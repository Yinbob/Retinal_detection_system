(() => {
  "use strict";
  const METHODS = { sift: "SIFT", sift_bidirectional: "双向 SIFT", hog: "HOG" };
  const PROTOCOLS = { vistel_course: "课程协议", fire_pair: "原始划分", fire_pair_exclude_self: "排除自身副本" };
  const STATUSES = { accepted: "接受", rejected: "拒绝", no_valid_candidate: "无有效候选" };
  const CATEGORIES = {
    false_accept_mismatch: "异眼误接受", cross_group_unconfirmed: "跨关系组未确认",
    threshold_marginal_rejection: "阈值边际误拒识", impostor_rejected: "异眼候选被拦截",
    same_eye_alternate_reference: "命中同眼其他参考图", exact_self_copy_hit: "命中自身副本",
    no_valid_candidate: "无有效几何候选",
  };
  const VIEWS = {
    identify: ["光学识别工作台", "选择一张眼底图，比较局部几何与全局梯度的身份检索结果。"],
    results: ["协议基准对比", "在固定评测协议下，比较三种方案的准确率、拒识与检索耗时。"],
    calibration: ["阈值标定与改进", "从代理开发集到冻结参数，追溯每一次算法选择的依据。"],
    errors: ["错误分析与审计", "查看异常样本、复现实验结果，并核对数据与模型完整性。"],
  };
  const state = {
    dataset: "vistel", protocol: "fire_pair_exclude_self", method: "all", source: "protocol",
    stage: "rgb", category: "ALL", file: null, previewUrl: null, config: null,
    summary: null, queries: [], busy: false, loadingQueries: false, results: [],
    reference: null, overlay: [], lines: true, queryRequest: 0, revision: 0,
    view: "identify", metricMode: "with_reject",
  };
  const $ = (id) => document.getElementById(id);
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "vars") Object.entries(value).forEach(([k, v]) => node.style.setProperty(k, String(v)));
      else node.setAttribute(key, value);
    }
    node.append(...children);
    return node;
  }
  const score = (n) => n == null ? "—" : Number(n).toFixed(3);
  const pct = (n) => n == null ? "—" : `${(Number(n) * 100).toFixed(2)}%`;
  const ms = (n) => n == null ? "—" : `${Number(n).toFixed(n < 1 ? 2 : 0)} ms`;
  const clamp = (n) => Math.max(0, Math.min(1, Number(n) || 0));
  const imageUrl = (id, stage = state.stage) => `/api/image?${new URLSearchParams({ dataset: state.dataset, id, stage })}`;
  const methodList = () => state.method === "all" ? Object.keys(METHODS) : [state.method];
  async function request(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    let body;
    try { body = await response.json(); } catch { throw new Error("服务返回了无法读取的数据，请刷新后重试"); }
    if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
    return body;
  }
  function message(box, title, text, error = false, retry) {
    const children = [el("strong", { text: title }), el("p", { text })];
    if (retry) {
      const button = el("button", { type: "button", class: "toggle-btn", text: "重新加载" });
      button.addEventListener("click", retry);
      children.push(button);
    }
    box.replaceChildren(el("div", { class: `empty-state${error ? " is-error" : ""}`, role: error ? "alert" : "status" }, children));
  }
  function segment(id, value) {
    for (const button of $(id).querySelectorAll("button")) {
      const on = button.dataset.value === value;
      button.classList.toggle("is-on", on);
      button.setAttribute("aria-checked", String(on));
      button.tabIndex = on ? 0 : -1;
    }
  }
  function bindSegment(id, key, after) {
    const group = $(id);
    group.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button || button.disabled || state[key] === button.dataset.value) return;
      state[key] = button.dataset.value;
      segment(id, state[key]);
      Promise.resolve(after()).catch(showError);
    });
    group.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      const buttons = [...group.querySelectorAll("button:not(:disabled)")];
      const index = buttons.indexOf(document.activeElement);
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1) + buttons.length) % buttons.length;
      event.preventDefault(); buttons[next].focus(); buttons[next].click();
    });
  }
  function updateRunButton() {
    $("runBtn").disabled = state.busy || state.loadingQueries || !state.config || (state.source === "upload" ? !state.file : !$("querySelect").value);
  }
  function showError(error) {
    message($("results"), "暂时无法完成操作", error.message, true);
  }
  function thresholdHint() {
    if (!state.config) return;
    $("thresholdHint").textContent = methodList().map((m) => `${METHODS[m]} ${score(state.config.methods[m]?.threshold)}`).join(" · ") + "。阈值已在代理协议上冻结。";
  }
  function setDisc(id, placeholder, src, alt) {
    const disc = $(id);
    let img = disc.querySelector("img");
    $(placeholder).hidden = Boolean(src);
    if (!src) { img?.remove(); return; }
    if (!img) {
      img = el("img", { alt, draggable: "false" });
      img.addEventListener("error", () => {
        img.remove(); $(placeholder).hidden = false; $(placeholder).textContent = "图像未能载入，请重新选择样本";
      });
      img.addEventListener("load", drawLines);
      disc.prepend(img);
    }
    img.alt = alt;
    if (img.getAttribute("src") !== src) img.src = src;
  }
  function refreshPreview() {
    const query = $("querySelect").value;
    const upload = state.source === "upload";
    setDisc("queryDisc", "previewPlaceholder", upload ? state.previewUrl : query ? imageUrl(query) : null, upload ? state.file?.name || "上传图像" : query);
    // Uploaded bytes are not stored as manifest images; preprocessing endpoints apply to protocol images only.
    for (const button of $("stageViewSeg").querySelectorAll("button")) button.disabled = upload && button.dataset.value !== "rgb";
    $("queryMeta").textContent = upload ? state.file?.name || "未选择图片" : query || "—";
    $("queryFootMeta").textContent = upload ? "本地上传 · 全参考库检索" : query ? `${state.dataset === "vistel" ? "Vistel" : "FIRE"} · ${PROTOCOLS[state.dataset === "vistel" ? "vistel_course" : state.protocol]}` : "等待载入查询样本";
    if (state.reference) setDisc("refDisc", "refPlaceholder", imageUrl(state.reference), state.reference);
    updateStageCaption();
  }
  function updateStageCaption() {
    const stages = { rgb: "RGB 原图", green: "绿通道", fov: "FOV 掩膜", clahe: "CLAHE 增强" };
    $("stageSubtitle").textContent = state.source === "upload"
      ? "上传查询显示原图；预处理透视适用于协议查询样本"
      : !["rgb", "green"].includes(state.stage)
        ? `${stages[state.stage]} · 处理坐标系预览，几何连线已隐藏`
        : state.overlay.length
          ? `${stages[state.stage]} · ${state.overlay.length} 组真实 RANSAC 内点对应${state.lines ? "" : "（连线已隐藏）"}`
          : "切换图像处理阶段；局部算法 Top-1 支持真实几何内点连线";
  }
  function resetResults() {
    state.revision += 1;
    state.results = []; state.reference = null; state.overlay = [];
    $("scanner").dataset.state = "idle";
    $("refScanner").dataset.state = "idle";
    setDisc("refDisc", "refPlaceholder", null, "");
    $("refMeta").textContent = "—";
    $("refRoleLabel").textContent = "最相似参考眼底 (Top-1 Reference)";
    $("refFootMeta").textContent = "执行比对后，可点击候选图切换参考图";
    $("bridgeBadge").className = "bridge-pill";
    $("bridgeBadge").textContent = "待比对";
    $("bridgeStat").textContent = "全参考库检索";
    $("stageSubtitle").textContent = "切换图像处理阶段，观察眼底结构与匹配结果";
    message($("results"), "准备开始身份比对", "选择查询样本后执行比对，查看各算法的得分、判定与 Top-5 候选。 ");
    drawLines(); updateRunButton();
  }
  function selectCandidate(method, result, candidate, button) {
    state.reference = candidate.image_id;
    const top = candidate.image_id === result.reference_id;
    state.overlay = top ? result.inlier_correspondences || [] : [];
    setDisc("refDisc", "refPlaceholder", imageUrl(state.reference), state.reference);
    $("refMeta").textContent = state.reference;
    $("refRoleLabel").textContent = `${METHODS[method]} · ${top ? "Top-1 参考图" : "候选参考图"}`;
    $("refFootMeta").textContent = `匹配分数 ${score(candidate.score)}${top && result.pair_stats?.n_inliers ? ` · ${result.pair_stats.n_inliers} 个几何内点` : ""}`;
    $("bridgeBadge").textContent = top ? STATUSES[result.status] || result.status : "候选预览";
    $("bridgeBadge").className = `bridge-pill ${top ? result.accepted ? "is-accepted" : "is-rejected" : ""}`;
    $("bridgeStat").textContent = `${METHODS[method]} · ${score(candidate.score)}`;
    $("refScanner").dataset.state = top ? result.accepted ? "accepted" : "rejected" : "idle";
    for (const item of document.querySelectorAll(".candidate")) { item.classList.remove("is-selected"); item.setAttribute("aria-pressed", "false"); }
    if (button) { button.classList.add("is-selected"); button.setAttribute("aria-pressed", "true"); }
    updateStageCaption();
    if (top && method === "hog") $("stageSubtitle").textContent = "HOG 使用全局梯度余弦相似度，不生成局部几何连线";
    drawLines();
  }
  function drawLines() {
    const svg = $("inlierSvg"); svg.replaceChildren();
    if (!state.lines || !["rgb", "green"].includes(state.stage) || state.source === "upload") return;
    const frame = $("dualOptics").getBoundingClientRect();
    if (!frame.width || !frame.height) return;
    svg.setAttribute("viewBox", `0 0 ${frame.width} ${frame.height}`);
    function point(id, x, y) {
      const disc = $(id).getBoundingClientRect();
      const img = $(id).querySelector("img");
      if (!img?.naturalWidth) return null;
      const scale = Math.min(disc.width / img.naturalWidth, disc.height / img.naturalHeight);
      const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
      return [disc.left - frame.left + (disc.width - w) / 2 + x * w, disc.top - frame.top + (disc.height - h) / 2 + y * h];
    }
    for (const p of state.overlay) {
      const q = point("queryDisc", p.qx, p.qy), r = point("refDisc", p.rx, p.ry);
      if (!q || !r) continue;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      for (const [key, value] of Object.entries({ x1: q[0], y1: q[1], x2: r[0], y2: r[1], class: "inlier-line" })) line.setAttribute(key, value);
      svg.append(line);
      for (const [cx, cy] of [q, r]) {
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        for (const [key, value] of Object.entries({ cx, cy, r: 2, class: "inlier-dot" })) dot.setAttribute(key, value);
        svg.append(dot);
      }
    }
  }
  function renderCard(method, result) {
    const candidates = result.top_candidates || [];
    const positives = new Set([...(result.positive_reference_ids || []), ...(result.confirmed_reference_ids || [])]);
    const domain = Math.max(result.threshold * 1.5, ...candidates.map((c) => c.score), 1);
    const accepted = result.status === "accepted";
    const rows = [["最相似参考图", result.reference_id || "—"], ["参考库", `${result.gallery_size} 张`], ["检索耗时", ms(result.elapsed_ms)]];
    if (result.designated_correct !== undefined) rows.push(["协议判定", result.designated_correct ? "命中正确参考" : result.confirmed_same_eye_correct ? "已确认同眼，非指定配对" : result.same_eye_status === "same_eye_unconfirmed" ? "同眼关系未确认" : result.status === "accepted" ? "未命中正确参考" : "未通过认证"]);
    if (method !== "hog" && result.pair_stats) rows.push(["几何内点 / 比率", `${result.pair_stats.n_inliers} / ${pct(result.pair_stats.inlier_ratio)}`]);
    const card = el("article", { class: "method-card" }, [
      el("div", { class: "method-head" }, [el("h3", { class: "method-title", text: METHODS[method] }), el("span", { class: `badge ${accepted ? "accepted" : result.status === "rejected" ? "rejected" : "failure"}`, text: STATUSES[result.status] || result.status })]),
      el("div", { class: "meter-block" }, [el("div", { class: "meter-labels" }, [el("span", { text: `分数 ${score(result.score)}` }), el("span", { text: `阈值 ${score(result.threshold)}` })]), el("div", { class: "meter", "aria-label": `得分 ${score(result.score)}，阈值 ${score(result.threshold)}` }, [el("span", { class: `meter-fill${accepted ? "" : " is-rejected"}`, vars: { "--v": clamp(result.score / domain) } }), el("span", { class: "meter-mark", vars: { "--pos": clamp(result.threshold / domain) } })])]),
      keyValues(rows),
    ]);
    if (candidates.length) {
      card.append(el("div", { class: "candidate-heading" }, [el("span", { text: `Top-${candidates.length} 候选 · 点击查看` }), el("span", { text: "绿点表示已知正确参考" })]));
      card.append(el("div", { class: "candidates" }, candidates.map((candidate, index) => {
        const button = el("button", { type: "button", class: `candidate${index === 0 ? " is-top" : ""}${positives.has(candidate.image_id) ? " is-positive" : ""}`, "aria-label": `${METHODS[method]} 候选 ${index + 1}，${candidate.image_id}，分数 ${score(candidate.score)}`, "aria-pressed": "false" }, [
          el("img", { src: candidate.thumb_url, alt: candidate.image_id, loading: "lazy", width: 96, height: 96 }),
          el("span", { class: "candidate-foot" }, [el("span", { text: candidate.image_id, title: candidate.image_id }), el("span", { class: "score", text: score(candidate.score) })]),
        ]);
        button.addEventListener("click", () => selectCandidate(method, result, candidate, button));
        return button;
      })));
    }
    if (result.diagnostic_note) card.append(el("p", { class: "diag-callout", text: result.diagnostic_note }));
    return card;
  }
  function keyValues(rows) {
    return el("dl", { class: "kv" }, rows.map(([k, v]) => el("div", {}, [el("dt", { text: k }), el("dd", { text: v })])));
  }
  async function runIdentification() {
    if (state.busy) return;
    if (state.source === "upload" && !state.file) return;
    if (state.source === "protocol" && !$("querySelect").value) return;
    const revision = state.revision;
    const query = { dataset: state.dataset, protocol: state.protocol, id: $("querySelect").value, source: state.source, file: state.file };
    const methods = methodList();
    state.busy = true; updateRunButton();
    $("runBtn").classList.add("is-busy");
    $("scanner").dataset.state = "busy";
    $("results").setAttribute("aria-busy", "true");
    const results = [];
    try {
      for (const [index, method] of methods.entries()) {
        $("runLabel").textContent = `比对中 ${index + 1}/${methods.length}`;
        message($("results"), `正在执行 ${METHODS[method]} 比对`, "正在读取特征并检索参考库，请稍候。切换样本会使本次结果失效。");
        const params = new URLSearchParams({ dataset: query.dataset, method });
        let result;
        if (query.source === "protocol") {
          params.set("id", query.id);
          if (query.dataset === "fire") params.set("protocol", query.protocol);
          result = await request(`/api/identify/protocol?${params}`, { method: "POST" });
        } else {
          result = await request(`/api/identify/upload?${params}`, { method: "POST", headers: { "Content-Type": query.file.type }, body: query.file });
        }
        if (revision !== state.revision) return;
        results.push({ method, result });
      }
      state.results = results;
      $("results").replaceChildren(...results.map(({ method, result }) => renderCard(method, result)));
      const accepted = results.filter(({ result }) => result.accepted).length;
      $("scanner").dataset.state = accepted === results.length ? "accepted" : accepted ? "mixed" : "rejected";
      const first = results.find(({ result }) => result.reference_id);
      if (first) selectCandidate(first.method, first.result, first.result.top_candidates.find((c) => c.image_id === first.result.reference_id), $("results").querySelector(".candidate"));
      else { $("bridgeBadge").textContent = "无有效候选"; $("bridgeStat").textContent = "可切换样本或算法后重试"; }
    } catch (error) {
      if (revision === state.revision) { $("scanner").dataset.state = "idle"; showError(error); }
    } finally {
      state.busy = false; $("runBtn").classList.remove("is-busy"); $("runLabel").textContent = "执行身份比对";
      $("results").setAttribute("aria-busy", "false"); updateRunButton();
    }
  }
  function renderQueryOptions(preferred) {
    const visible = state.queries.filter((q) => state.dataset !== "fire" || state.category === "ALL" || q.category === state.category);
    $("querySelect").replaceChildren(...visible.map((q) => el("option", { value: q.image_id, text: q.subject_id != null ? `${q.image_id} · ${q.eye === "left" ? "左眼" : "右眼"}` : `${q.image_id} · ${q.category} 组` })));
    if (visible.some((q) => q.image_id === preferred)) $("querySelect").value = preferred;
    $("queryCountBadge").textContent = `${visible.length} 张`;
    $("querySelect").disabled = !visible.length;
    resetResults(); refreshPreview();
  }
  function renderPresets() {
    const cases = Object.values(state.summary?.report?.error_analysis?.by_run || {}).flatMap((r) => r.cases || []);
    const examples = [];
    const seen = new Set();
    for (const c of cases.filter((c) => c.dataset === state.dataset && (state.dataset !== "fire" || c.protocol === state.protocol))) {
      if (seen.has(c.error_category)) continue;
      seen.add(c.error_category); examples.push(c);
      if (examples.length === 3) break;
    }
    $("presetButtons").replaceChildren(...examples.map((c) => {
      const button = el("button", { type: "button", class: "preset-btn" }, [el("span", { text: c.query_id }), el("span", { class: "preset-tag", text: `${METHODS[c.method]} · ${CATEGORIES[c.error_category] || c.error_category}` })]);
      button.addEventListener("click", () => {
        state.method = c.method; segment("methodSeg", state.method); thresholdHint();
        state.category = "ALL"; updateCategory(); renderQueryOptions(c.query_id);
      });
      return button;
    }));
    $("presetButtons").parentElement.hidden = !examples.length;
  }
  function updateCategory() {
    for (const button of $("fireCatFilter").querySelectorAll("button")) {
      const category = button.dataset.cat;
      button.classList.toggle("is-on", state.category === category);
      button.setAttribute("aria-pressed", String(state.category === category));
      const count = state.queries.filter((q) => category === "ALL" || q.category === category).length;
      button.textContent = `${{ ALL: "全部", S: "S 高重叠", P: "P 小重叠", A: "A 病变" }[category]} (${count})`;
    }
  }
  async function loadQueries() {
    const ticket = ++state.queryRequest;
    state.loadingQueries = true;
    $("querySelect").disabled = true; updateRunButton();
    try {
      const data = await request(`/api/protocol-queries?dataset=${state.dataset}`);
      if (ticket !== state.queryRequest) return;
      state.queries = data.queries; state.category = "ALL";
      $("protocolField").hidden = state.dataset !== "fire" || state.source === "upload";
      $("fireCatFilter").hidden = state.dataset !== "fire";
      updateContext();
      updateCategory(); renderQueryOptions(); renderPresets();
    } catch (error) {
      if (ticket === state.queryRequest) { state.queries = []; $("querySelect").replaceChildren(); message($("results"), "样本加载失败", error.message, true, loadQueries); }
    } finally {
      if (ticket === state.queryRequest) { state.loadingQueries = false; updateRunButton(); }
    }
  }
  function setFile(file) {
    if (file && (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 20 * 1024 * 1024 || !file.size)) {
      $("uploadError").textContent = "请选择有效的 JPG 或 PNG 图片，大小不超过 20 MB。";
      $("fileInput").value = ""; return;
    }
    $("uploadError").textContent = "";
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file; state.previewUrl = file ? URL.createObjectURL(file) : null;
    $("dropText").textContent = file ? file.name : "选择或拖入眼底彩照";
    $("dropzone").classList.toggle("has-file", Boolean(file));
    $("clearFileBtn").hidden = !file;
    if (state.source === "upload") { resetResults(); refreshPreview(); }
    updateRunButton();
  }
  function setSource() {
    const upload = state.source === "upload";
    $("protocolPanel").hidden = upload; $("uploadPanel").hidden = !upload;
    $("protocolField").hidden = upload || state.dataset !== "fire";
    if (upload) { state.stage = "rgb"; segment("stageViewSeg", "rgb"); }
    resetResults(); refreshPreview();
  }
  function table(headers, rows) {
    return el("table", {}, [el("thead", {}, [el("tr", {}, headers.map((text, i) => el("th", { text, scope: "col", class: i ? "num" : "" })))]), el("tbody", {}, rows.map((row) => el("tr", {}, row.map((text, i) => el("td", { text, class: i ? "num" : "" }))))) ]);
  }
  function bar(label, primary, secondary) {
    const track = (value, thin) => el("div", { class: `bar-track${thin ? " thin" : ""}` }, [el("span", { class: `bar-fill${thin ? " is-lower" : ""}`, vars: { "--v": clamp(value) } })]);
    return el("div", { class: "bar-row" }, [el("span", { class: "bar-label", text: label, title: label }), el("div", { class: "bar-stack" }, [track(primary), ...(secondary != null ? [track(secondary, true)] : [])]), el("span", { class: "bar-value", text: `${pct(primary)}${secondary == null ? "" : ` / ${pct(secondary)}`}` })]);
  }
  function renderCharts() {
    const report = state.summary?.report;
    if (!report) return;
    const top1 = $("metricModeSeg").querySelector('[aria-checked="true"]').dataset.value === "top1";
    const primary = top1 ? "top1_accuracy" : "accuracy", secondary = top1 ? "top1_confirmed_same_eye_accuracy" : "confirmed_same_eye_accuracy";
    const label = (row) => `${row.dataset === "vistel" ? "Vistel" : "FIRE"} · ${PROTOCOLS[row.protocol]} · ${METHODS[row.method]}`;
    $("accuracyChart").replaceChildren(...report.rows.map((row) => bar(label(row), row[primary], row[secondary])));
    $("categoryBody").replaceChildren(...(report.category_rows || []).map((row) => bar(`${PROTOCOLS[row.protocol]} · ${METHODS[row.method]} · ${row.category} 组`, row[primary], row[secondary])));
    for (const id of ["accuracyChart", "categoryBody"]) { $(id).classList.remove("muted"); $(id).classList.add("chart", "is-in"); }
  }
  function renderBenchmarks(report) {
    const rows = report.rows || [];
    const sift = rows.find((r) => r.dataset === "vistel" && r.method === "sift");
    const hog = rows.find((r) => r.dataset === "vistel" && r.method === "hog");
    const fire = rows.find((r) => r.protocol === "fire_pair_exclude_self" && r.method === "sift");
    const kpis = [
      ["Vistel · SIFT 准确率", pct(sift?.accuracy), sift ? `${sift.correct} / ${sift.queries} 查询命中 · ${sift.rejected} 拒识` : "暂无实验结果", "accent-text"],
      ["Vistel · HOG 准确率", pct(hog?.accuracy), hog ? `${hog.correct} / ${hog.queries} 查询命中 · ${hog.rejected} 拒识` : "暂无实验结果", "warn-text"],
      ["FIRE · SIFT 已确认同眼", pct(fire?.confirmed_same_eye_accuracy), `排除自身副本 · Top-1 ${pct(fire?.top1_confirmed_same_eye_accuracy)}`, "info-text"],
      ["Vistel · 检索中位耗时", `${ms(hog?.median_query_ms)} / ${ms(sift?.median_query_ms)}`, "HOG / SIFT · 以当前实验报告为准", ""],
    ];
    $("benchmarkKpis").replaceChildren(...kpis.map(([label, value, note, color]) => el("div", { class: "panel kpi-item" }, [el("span", { class: "kpi-label", text: label }), el("strong", { class: `kpi-value ${color}`, text: value }), el("span", { class: "kpi-note", text: note })])));
    $("metricsTable").querySelector("tbody").replaceChildren(...rows.map((r) => el("tr", {}, [r.dataset === "vistel" ? "Vistel" : "FIRE", PROTOCOLS[r.protocol], METHODS[r.method], `${r.top1_correct}/${r.queries}`, pct(r.top1_accuracy), `${r.correct}/${r.queries}`, pct(r.accuracy), r.confirmed_same_eye_accuracy == null ? "—" : `${pct(r.top1_confirmed_same_eye_accuracy)} / ${pct(r.confirmed_same_eye_accuracy)}`, r.rejected, r.failures, ms(r.median_query_ms)].map((text, i) => el("td", { text, class: i >= 3 ? "num" : "" })))));
    const registration = rows.filter((r) => r.registration_analysis && r.protocol === "fire_pair_exclude_self");
    $("registrationBody").replaceChildren(registration.length ? table(["方法", "已评估配对", "控制点", "中位 px", "P95 px", "最大 px"], registration.map((r) => {
      const s = r.registration_analysis; return [METHODS[r.method], `${s.evaluated_pairs}/${s.eligible_pairs}`, s.control_points, s.median_error_px.toFixed(2), s.p95_error_px.toFixed(2), s.max_error_px.toFixed(2)];
    })) : el("p", { class: "muted", text: "暂无配准误差数据" }));
    renderCharts();
  }
  function renderCalibration(calibration) {
    if (!calibration) {
      for (const id of ["calibrationBody", "improvementBody", "searchGridWrap"]) message($(id), "暂无标定结果", "完成代理协议标定后，此处会显示冻结阈值与参数搜索记录。");
      return;
    }
    const methods = Object.entries(calibration.methods || {});
    const threshold = (s) => s.threshold.threshold == null ? `无有限拒识阈值 · 最低内点 ${s.config.min_inliers ?? "—"}` : score(s.threshold.threshold);
    $("calibrationBody").replaceChildren(table(["方法", "标定阈值", "经验 FAR", "经验 FRR", "代理 Top-1"], methods.map(([m, d]) => [METHODS[m] || m, threshold(d.selected), pct(d.selected.threshold.empirical_far), pct(d.selected.threshold.empirical_frr), `${d.selected.top1_correct}/33`])));
    const base = calibration.methods.sift?.selected, bi = calibration.methods.sift?.bidirectional_candidate;
    if (base && bi) {
      $("improvementBody").replaceChildren(el("div", { class: "imp-metrics" }, [["单向匹配 · 经验 FAR", base], ["双向一致性 · 经验 FAR", bi]].map(([label, data]) => el("div", { class: "imp-stat" }, [el("span", { class: "imp-stat-title", text: label }), el("strong", { class: "imp-stat-val", text: pct(data.threshold.empirical_far) }), el("span", { class: "imp-stat-sub", text: `代理 Top-1 ${data.top1_correct}/33 · FRR ${pct(data.threshold.empirical_frr)}` })]))), el("p", { class: "muted small", text: `双向候选${calibration.methods.sift.bidirectional_adopted ? "已被采用为基础配置" : "作为独立对照方案保留"}。双向约束减少偶然匹配，也可能减少有效几何候选。` }));
    } else message($("improvementBody"), "暂无双向对照", "当前标定文件未包含双向候选记录。");
    const searches = methods.flatMap(([m, d]) => [...(d.search_candidates || []).map((s) => [m, "参数搜索", s]), ...(d.min_inlier_candidates || []).map((s) => [m, "最低内点搜索", s])]);
    $("searchGridWrap").replaceChildren(table(["方法 / 阶段", "比值", "RANSAC px", "最低内点", "阈值", "FAR", "FRR", "Top-1"], searches.map(([m, phase, s]) => [`${METHODS[m]} · ${phase}`, s.config.ratio ?? "—", s.config.ransac_px ?? "—", s.config.min_inliers ?? "—", threshold(s), pct(s.threshold.empirical_far), pct(s.threshold.empirical_frr), `${s.top1_correct}/33`])));
  }
  function allCases() { return Object.values(state.summary?.report?.error_analysis?.by_run || {}).flatMap((r) => r.cases || []); }
  function renderErrors() {
    const run = $("errorRunFilter").value, category = $("errorCatFilter").value;
    const cases = allCases().filter((c) => (run === "ALL" || c.run === run) && (category === "ALL" || c.error_category === category));
    $("errorCaseCount").textContent = `${cases.length} 个已收录样本`;
    $("errorList").replaceChildren(...cases.map((c, index) => {
      const button = el("button", { type: "button", class: `error-item${index === 0 ? " is-active" : ""}`, "aria-pressed": String(index === 0) }, [el("span", { class: "error-item-top" }, [el("span", { text: c.query_id }), el("span", { text: METHODS[c.method] })]), el("span", { class: "error-item-sub" }, [el("span", { text: CATEGORIES[c.error_category] || c.error_category }), el("span", { text: PROTOCOLS[c.protocol] })])]);
      button.addEventListener("click", () => {
        for (const b of $("errorList").querySelectorAll("button")) { b.classList.toggle("is-active", b === button); b.setAttribute("aria-pressed", String(b === button)); }
        renderErrorDetail(c);
      });
      return button;
    }));
    if (cases.length) renderErrorDetail(cases[0]);
    else { message($("errorList"), "没有匹配的样本", "试试其他实验运行或错误类别。"); message($("errorDetail"), "等待选择案例", "切换筛选条件以检视异常与边界样本。"); }
  }
  function renderErrorDetail(c) {
    const replay = el("button", { type: "button", class: "toggle-btn", text: "载入识别台复现", disabled: state.busy ? "" : null });
    replay.addEventListener("click", async () => {
      if (state.busy) return;
      try {
        state.dataset = c.dataset; state.protocol = c.protocol; state.source = "protocol"; state.method = c.method;
        for (const [id, key] of [["datasetSeg", "dataset"], ["protocolSeg", "protocol"], ["sourceSeg", "source"], ["methodSeg", "method"]]) segment(id, state[key]);
        setSource(); thresholdHint(); switchView("identify");
        await loadQueries();
        if (!state.queries.some((q) => q.image_id === c.query_id)) throw new Error("该查询样本不在当前协议中");
        renderQueryOptions(c.query_id);
        await runIdentification();
      } catch (error) { showError(error); }
    });
    const images = [["查询图", c.query_id], ["返回参考", c.reference_id], ["指定正确参考", c.positive_reference_id]].map(([label, id]) => el("figure", {}, [id ? el("img", { src: `/api/image?${new URLSearchParams({ dataset: c.dataset, id })}`, alt: `${label} ${id}`, loading: "lazy", width: 220, height: 220 }) : el("div", { class: "missing-image", text: "无有效候选" }), el("figcaption", { text: `${label} · ${id || "—"}` })]));
    const content = [el("div", { class: "error-detail-head" }, [el("h3", { text: `${c.query_id} · ${CATEGORIES[c.error_category] || c.error_category}` }), replay]), el("p", { class: "muted small", text: c.diagnostic_note }), el("div", { class: "case-images" }, images), keyValues([["匹配分数", score(c.score)], ["冻结阈值", score(c.threshold)], ["Top-1 / Top-2 间隔", score(c.score_margin)], ["认证状态", STATUSES[c.status] || c.status]])];
    if (c.panel_index != null) content.push(el("details", { class: "panel-disclosure" }, [el("summary", { text: "展开导出的匹配诊断图" }), el("img", { class: "error-panel-img", src: `/api/error-panel?${new URLSearchParams({ run: c.run, index: c.panel_index })}`, alt: `${c.query_id} 匹配诊断图`, loading: "lazy" })]));
    $("errorDetail").replaceChildren(...content);
  }
  function renderAudit(audit, calibration) {
    const datasets = audit?.datasets || {};
    const blocks = Object.entries(datasets).map(([name, data]) => {
      const s = data.summary;
      const rows = name === "vistel" ? [["图像", s.images], ["身份 / 眼", `${s.subjects} / ${s.eyes}`], ["查询 / 参考", `${s.queries} / ${s.gallery}`], ["像素重复", s.pixel_hash_duplicates]] : [["图像 / 独立内容", `${s.images} / ${s.unique_content}`], ["同眼关系组", s.relation_groups], ["含参考副本的查询", s.queries_with_exact_gallery_copy], ["排除的自身配对", s.blocked_self_copy_combinations]];
      return el("section", { class: "audit-block" }, [el("h3", { text: name === "vistel" ? "Vistel 数据契约" : "FIRE 重复内容审计" }), keyValues(rows), el("details", { class: "panel-disclosure" }, [el("summary", { text: "查看源文件 SHA-256" }), el("code", { class: "hash", text: data.archive_sha256 })])]);
    });
    const model = calibration?.superretina;
    blocks.push(el("section", { class: "audit-block" }, [el("h3", { text: "SuperRetina 模型完整性" }), el("span", { class: `badge ${model?.status === "skipped" ? "failure" : "info"}`, text: model?.status === "skipped" ? "未启用" : model?.status || "暂无审计结果" }), el("p", { class: "muted small", text: model?.status === "skipped" ? "官方源码或预训练权重缺失。SIFT 与 HOG 可独立运行。" : "模型状态以当前审计文件为准。" }), keyValues([["官方源码", model ? model.source_present ? "已找到" : "缺失" : "—"], ["预训练权重", model ? model.weights_present ? "已找到" : "缺失" : "—"], ["随机初始化", model ? model.random_initialization_used ? "使用" : "未使用" : "—"]])]));
    $("auditBody").replaceChildren(...blocks);
    const checks = Object.entries(audit?.checks || {});
    if (checks.length) $("auditBody").append(el("p", { class: "audit-summary", text: `${checks.filter(([, v]) => v).length} / ${checks.length} 项数据与协议检查通过` }));
  }
  function renderDiagnostics(report) {
    const byRun = report.error_analysis?.by_run || {};
    $("errorRunFilter").replaceChildren(el("option", { value: "ALL", text: "全部实验运行" }), ...report.rows.map((r) => el("option", { value: r.run, text: `${r.dataset === "vistel" ? "Vistel" : "FIRE"} · ${PROTOCOLS[r.protocol]} · ${METHODS[r.method]}` })));
    const counts = {};
    for (const r of Object.values(byRun)) for (const [key, value] of Object.entries(r.category_counts || {})) counts[key] = (counts[key] || 0) + value;
    const findings = [
      ["异眼误接受", counts.false_accept_mismatch || 0, "已归类的异眼误接受；可在下方切换实验运行，检视查询与返回参考。", "false_accept_mismatch"],
      ["跨关系组返回", counts.cross_group_unconfirmed || 0, "FIRE 跨组结果表示同眼关系未确认，不能直接作为确定的异眼真值。", "cross_group_unconfirmed"],
      ["阈值边际误拒识", counts.threshold_marginal_rejection || 0, "最相似候选未通过冻结阈值；结合内点数与得分间隔检查边界样本。", "threshold_marginal_rejection"],
      ["命中自身副本", counts.exact_self_copy_hit || 0, "原始划分含相同源图，比较排除自身副本协议可观察其对结果的影响。", "exact_self_copy_hit"],
    ];
    $("findingsGrid").replaceChildren(...findings.map(([title, count, text, category]) => {
      const button = el("button", { type: "button", class: "finding-item" }, [el("span", { class: "finding-head" }, [el("span", { class: "finding-title", text: title }), el("span", { class: "meta-pill", text: `${count} 次` })]), el("span", { class: "finding-body", text })]);
      button.addEventListener("click", () => { $("errorCatFilter").value = category; $("errorRunFilter").value = "ALL"; renderErrors(); $("errorInspector").scrollIntoView({ block: "start" }); });
      return button;
    }));
    renderErrors();
  }
  async function loadSummary() {
    try {
      state.summary = await request("/api/summary");
      if (state.summary.report) { renderBenchmarks(state.summary.report); renderDiagnostics(state.summary.report); }
      else {
        $("benchmarkKpis").replaceChildren(); $("metricsTable").querySelector("tbody").replaceChildren();
        for (const id of ["accuracyChart", "categoryBody", "registrationBody", "findingsGrid", "errorList", "errorDetail"]) message($(id), "暂无实验报告", "完成正式评测并生成报告后，在此查看结果。");
      }
      renderCalibration(state.summary.calibration); renderAudit(state.summary.data_audit, state.summary.calibration); renderPresets();
      updateContext();
    } catch (error) {
      for (const id of ["accuracyChart", "categoryBody", "registrationBody", "calibrationBody", "improvementBody", "searchGridWrap", "findingsGrid", "errorList", "errorDetail", "auditBody"]) message($(id), "实验数据加载失败", error.message, true, loadSummary);
      $("benchmarkKpis").replaceChildren();
    }
  }
  function moveTabThumb() {
    const tab = document.querySelector(".tab.is-active");
    $("tabThumb").style.width = `${tab.offsetWidth}px`;
    $("tabThumb").style.transform = `translateX(${tab.offsetLeft}px)`;
    $("tabThumb").classList.add("is-ready");
  }
  function updateContext() {
    const info = state.config?.datasets[state.dataset];
    $("datasetContext").textContent = state.view === "identify"
      ? info ? `${state.dataset === "vistel" ? "Vistel" : "FIRE"} · ${info.queries} 张查询 / ${info.gallery} 张参考` : "正在读取数据集"
      : state.view === "results"
        ? `${state.summary?.report?.rows?.length || 0} 组实验 · Vistel / FIRE`
        : state.view === "calibration" ? "Vistel · 33 / 125 代理协议" : `${allCases().length} 个已收录样本 · 数据审计`;
  }
  function switchView(view) {
    if (!VIEWS[view]) view = "identify";
    state.view = view;
    for (const tab of document.querySelectorAll(".tab")) {
      const on = tab.dataset.view === view;
      tab.classList.toggle("is-active", on); tab.setAttribute("aria-selected", String(on)); tab.tabIndex = on ? 0 : -1;
    }
    for (const name of Object.keys(VIEWS)) { $(`view-${name}`).hidden = name !== view; $(`view-${name}`).classList.toggle("is-active", name === view); }
    $("pageTitle").textContent = VIEWS[view][0]; $("pageDescription").textContent = VIEWS[view][1];
    history.replaceState(null, "", `#${view}`); updateContext(); moveTabThumb(); requestAnimationFrame(drawLines);
    document.querySelector(".tab.is-active").scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  function bindUpload() {
    $("fileInput").addEventListener("change", (event) => setFile(event.target.files[0] || null));
    $("clearFileBtn").addEventListener("click", () => { $("fileInput").value = ""; setFile(null); });
    for (const type of ["dragenter", "dragover", "dragleave", "drop"]) $("dropzone").addEventListener(type, (event) => {
      event.preventDefault(); $("dropzone").classList.toggle("is-drag", type === "dragenter" || type === "dragover");
      if (type === "drop" && event.dataTransfer.files.length) setFile(event.dataTransfer.files[0]);
    });
  }
  async function connect() {
    $("serviceText").textContent = "连接中";
    try {
      state.config = await request("/api/datasets");
      $("serviceStatus").className = "status is-ok"; $("serviceText").textContent = "本地服务在线";
      thresholdHint(); await loadQueries();
    } catch (error) {
      state.config = null; updateRunButton();
      $("serviceStatus").className = "status is-error"; $("serviceText").textContent = "服务未连接";
      message($("results"), "无法连接本地服务", "请确认本地服务已启动，然后重新连接。", true, connect);
    }
  }
  async function init() {
    for (const tab of document.querySelectorAll(".tab")) {
      tab.id = `tab-${tab.dataset.view}`; tab.setAttribute("aria-controls", `view-${tab.dataset.view}`);
      const panel = $(`view-${tab.dataset.view}`); panel.setAttribute("role", "tabpanel"); panel.setAttribute("aria-labelledby", tab.id);
      tab.addEventListener("click", () => switchView(tab.dataset.view));
    }
    document.querySelector(".tabs").addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const tabs = [...document.querySelectorAll(".tab")], index = tabs.indexOf(document.activeElement);
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowLeft" ? -1 : 1) + tabs.length) % tabs.length;
      event.preventDefault(); tabs[next].focus(); tabs[next].click();
    });
    bindSegment("datasetSeg", "dataset", () => { resetResults(); return loadQueries(); });
    bindSegment("methodSeg", "method", () => { resetResults(); thresholdHint(); });
    bindSegment("sourceSeg", "source", setSource);
    bindSegment("protocolSeg", "protocol", () => { resetResults(); refreshPreview(); renderPresets(); });
    bindSegment("stageViewSeg", "stage", refreshPreview);
    bindSegment("metricModeSeg", "metricMode", renderCharts);
    for (const [id, key] of [["datasetSeg", "dataset"], ["methodSeg", "method"], ["sourceSeg", "source"], ["protocolSeg", "protocol"], ["stageViewSeg", "stage"]]) segment(id, state[key]);
    $("toggleLinesBtn").setAttribute("aria-pressed", "true");
    $("toggleLinesBtn").addEventListener("click", () => { state.lines = !state.lines; $("toggleLinesBtn").classList.toggle("is-on", state.lines); $("toggleLinesBtn").setAttribute("aria-pressed", String(state.lines)); updateStageCaption(); drawLines(); });
    $("querySelect").addEventListener("change", () => { resetResults(); refreshPreview(); });
    $("fireCatFilter").addEventListener("click", (event) => { const button = event.target.closest("button"); if (!button) return; state.category = button.dataset.cat; updateCategory(); renderQueryOptions(); });
    $("runBtn").addEventListener("click", runIdentification);
    $("errorRunFilter").addEventListener("change", renderErrors); $("errorCatFilter").addEventListener("change", renderErrors);
    bindUpload(); switchView(location.hash.slice(1) || "identify");
    window.addEventListener("hashchange", () => { if (VIEWS[location.hash.slice(1)]) switchView(location.hash.slice(1)); });
    window.addEventListener("resize", () => { moveTabThumb(); drawLines(); });
    new ResizeObserver(drawLines).observe($("dualOptics"));
    resetResults(); await connect(); await loadSummary();
  }
  init().catch(showError);
})();
