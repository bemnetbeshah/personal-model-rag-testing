const form = document.querySelector("#question-form");
const question = document.querySelector("#question");
const chatModel = document.querySelector("#chat-model");
const embeddingModel = document.querySelector("#embedding-model");
const button = form.querySelector("button");
const judgeEnabled = document.querySelector("#judge-enabled");
const judgeProvider = document.querySelector("#judge-provider");
const judgeModel = document.querySelector("#judge-model");
const judgeAnswer = document.querySelector("#judge-answer");
const judgeState = document.querySelector("#judge-state-label");
const judgeModelLabel = document.querySelector("#judge-model-label");

const lanes = {
  left: {
    provider: document.querySelector("#left-provider"),
    model: document.querySelector("#left-model"),
    answer: document.querySelector("#left-answer"),
    state: document.querySelector("#left-state-label"),
    timer: document.querySelector("#left-timer"),
  },
  right: {
    provider: document.querySelector("#right-provider"),
    model: document.querySelector("#right-model"),
    answer: document.querySelector("#right-answer"),
    state: document.querySelector("#right-state-label"),
    timer: document.querySelector("#right-timer"),
  },
};

let modelOptions = null;
let timerId = null;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let listItems = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) return;
    blocks.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 2, 4);
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      return;
    }

    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  return blocks.join("") || "<p></p>";
}

function setMarkdown(element, markdown) {
  element.innerHTML = renderMarkdown(markdown);
}

function getProviderConfig(providerId) {
  return modelOptions.providers.find((provider) => provider.id === providerId);
}

function setBusy(isBusy) {
  button.disabled = isBusy;
  question.disabled = isBusy;
  Object.values(lanes).forEach((lane) => {
    lane.provider.disabled = isBusy;
    lane.model.disabled = isBusy;
  });
  judgeEnabled.disabled = isBusy;
  judgeProvider.disabled = isBusy || !judgeEnabled.checked;
  judgeModel.disabled = isBusy || !judgeEnabled.checked;
}

function startTimer() {
  const start = performance.now();
  timerId = window.setInterval(() => {
    const elapsed = `${((performance.now() - start) / 1000).toFixed(1).padStart(4, "0")}s`;
    lanes.left.timer.textContent = elapsed;
    lanes.right.timer.textContent = elapsed;
  }, 100);
}

function stopTimer() {
  window.clearInterval(timerId);
  timerId = null;
}

function renderProviderOptions(laneName) {
  const lane = lanes[laneName];
  lane.provider.innerHTML = "";

  modelOptions.providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.enabled
      ? provider.label
      : `${provider.label} unavailable`;
    option.disabled = !provider.enabled;
    lane.provider.append(option);
  });
}

function renderProviderSelect(selectElement) {
  selectElement.innerHTML = "";
  modelOptions.providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.enabled
      ? provider.label
      : `${provider.label} unavailable`;
    option.disabled = !provider.enabled;
    selectElement.append(option);
  });
}

function renderModelOptions(laneName, preferredModel = null) {
  const lane = lanes[laneName];
  const provider = getProviderConfig(lane.provider.value);
  lane.model.innerHTML = "";

  provider.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    lane.model.append(option);
  });

  if (preferredModel && provider.models.includes(preferredModel)) {
    lane.model.value = preferredModel;
  }
}

function renderModelSelect(providerElement, modelElement, preferredModel = null) {
  const provider = getProviderConfig(providerElement.value);
  modelElement.innerHTML = "";
  provider.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelElement.append(option);
  });

  if (preferredModel && provider.models.includes(preferredModel)) {
    modelElement.value = preferredModel;
  }
}

function defaultRightChoice() {
  const anthropic = getProviderConfig("anthropic");
  if (anthropic?.enabled) {
    return { provider: "anthropic", model: anthropic.models[0] };
  }

  const ollama = getProviderConfig("ollama");
  const fallback =
    ollama.models.find((model) => model !== modelOptions.default.model) ||
    ollama.models[0];
  return { provider: "ollama", model: fallback };
}

function defaultJudgeChoice() {
  const anthropic = getProviderConfig("anthropic");
  if (anthropic?.enabled) {
    return { provider: "anthropic", model: anthropic.models[0] };
  }

  return modelOptions.default;
}

function renderModelControls() {
  Object.keys(lanes).forEach((laneName) => {
    renderProviderOptions(laneName);
  });

  lanes.left.provider.value = modelOptions.default.provider;
  renderModelOptions("left", modelOptions.default.model);

  const right = defaultRightChoice();
  lanes.right.provider.value = right.provider;
  renderModelOptions("right", right.model);

  Object.entries(lanes).forEach(([laneName, lane]) => {
    lane.provider.addEventListener("change", () => renderModelOptions(laneName));
  });

  renderProviderSelect(judgeProvider);
  const judge = defaultJudgeChoice();
  judgeProvider.value = judge.provider;
  renderModelSelect(judgeProvider, judgeModel, judge.model);
  judgeProvider.addEventListener("change", () => {
    renderModelSelect(judgeProvider, judgeModel);
    updateJudgeControls();
  });
  judgeModel.addEventListener("change", updateJudgeControls);
  judgeEnabled.addEventListener("change", updateJudgeControls);
  updateJudgeControls();
}

function updateJudgeControls() {
  judgeProvider.disabled = !judgeEnabled.checked;
  judgeModel.disabled = !judgeEnabled.checked;
  judgeState.textContent = judgeEnabled.checked ? "Judge ready" : "Judge off";
  judgeModelLabel.textContent = judgeEnabled.checked
    ? `${judgeProvider.value} / ${judgeModel.value}`
    : "No judge selected";
}

async function loadHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) return;

  const data = await response.json();
  chatModel.textContent = "Compare mode";
  embeddingModel.textContent = data.embedding_model;
  modelOptions = data.model_options;
  renderModelControls();
}

function selectedModel(lane) {
  return {
    provider: lane.provider.value,
    model: lane.model.value,
  };
}

function selectedJudge() {
  return {
    enabled: judgeEnabled.checked,
    provider: judgeProvider.value,
    model: judgeModel.value,
  };
}

function setLanePending(laneName) {
  const lane = lanes[laneName];
  lane.answer.classList.remove("is-error");
  lane.raw = "";
  lane.state.textContent = `${laneName} thinking`;
  setMarkdown(lane.answer, `Asking \`${lane.model.value}\`...`);
}

function setLaneResult(laneName, result) {
  const lane = lanes[laneName];
  lane.state.textContent = result.provider;
  lane.raw = result.answer;
  setMarkdown(lane.answer, result.answer);
}

function setLaneError(laneName, message) {
  const lane = lanes[laneName];
  lane.state.textContent = `${laneName} error`;
  setMarkdown(lane.answer, message);
  lane.answer.classList.add("is-error");
}

function appendLaneDelta(laneName, text) {
  const lane = lanes[laneName];
  lane.raw = `${lane.raw || ""}${text}`;
  setMarkdown(lane.answer, lane.raw);
  lane.answer.scrollTop = lane.answer.scrollHeight;
}

function setJudgeMarkdown(markdown) {
  judgeAnswer.raw = markdown;
  setMarkdown(judgeAnswer, markdown);
}

function appendJudgeDelta(text) {
  judgeAnswer.raw = `${judgeAnswer.raw || ""}${text}`;
  setMarkdown(judgeAnswer, judgeAnswer.raw);
  judgeAnswer.scrollTop = judgeAnswer.scrollHeight;
}

async function readEventStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();

    events.forEach((eventBlock) => {
      const dataLine = eventBlock
        .split("\n")
        .find((line) => line.startsWith("data: "));
      if (!dataLine) return;
      handleStreamEvent(JSON.parse(dataLine.slice(6)));
    });
  }
}

function handleStreamEvent(event) {
  if (event.type === "setup") {
    setMarkdown(lanes.left.answer, event.message);
    setMarkdown(lanes.right.answer, event.message);
    return;
  }

  if (event.type === "context_ready") {
    embeddingModel.textContent = event.embedding_model;
    return;
  }

  if (event.type === "start") {
    lanes[event.lane].state.textContent = event.provider;
    return;
  }

  if (event.type === "delta") {
    appendLaneDelta(event.lane, event.text);
    return;
  }

  if (event.type === "done") {
    setLaneResult(event.lane, event);
    return;
  }

  if (event.type === "error") {
    setLaneError(event.lane, event.error);
    return;
  }

  if (event.type === "judge_start") {
    judgeState.textContent = "Judging";
    judgeModelLabel.textContent = `${event.provider} / ${event.chat_model}`;
    setJudgeMarkdown(`Asking \`${event.chat_model}\` to judge...`);
    return;
  }

  if (event.type === "judge_delta") {
    appendJudgeDelta(event.text);
    return;
  }

  if (event.type === "judge_done") {
    judgeState.textContent = "Judged";
    judgeModelLabel.textContent = `${event.provider} / ${event.chat_model}`;
    setJudgeMarkdown(event.verdict);
    return;
  }

  if (event.type === "judge_error") {
    judgeState.textContent = "Judge error";
    judgeAnswer.classList.add("is-error");
    setJudgeMarkdown(event.error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = question.value.trim();
  if (!value) {
    setLaneError("left", "Enter a question about the paper.");
    setLaneError("right", "Enter a question about the paper.");
    return;
  }

  setBusy(true);
  startTimer();
  setLanePending("left");
  setLanePending("right");
  judgeAnswer.classList.remove("is-error");
  judgeAnswer.raw = "";
  judgeState.textContent = judgeEnabled.checked ? "Judge waiting" : "Judge off";
  judgeModelLabel.textContent = judgeEnabled.checked
    ? `${judgeProvider.value} / ${judgeModel.value}`
    : "No judge selected";
  setJudgeMarkdown(judgeEnabled.checked
    ? `Waiting for both answers, then asking ${judgeModel.value} to judge...`
    : "Judging disabled for this comparison.");

  try {
    const response = await fetch("/api/compare-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: value,
        models: [selectedModel(lanes.left), selectedModel(lanes.right)],
        judge: selectedJudge(),
      }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Compare request failed.");
    }

    await readEventStream(response);
    chatModel.textContent = `${lanes.left.model.value} / ${lanes.right.model.value}`;
    if (!judgeEnabled.checked) {
      judgeState.textContent = "Judge off";
      setJudgeMarkdown("Judging disabled for this comparison.");
    }
  } catch (error) {
    setLaneError("left", error.message);
    setLaneError("right", error.message);
    judgeState.textContent = "Judge error";
    setJudgeMarkdown(error.message);
    judgeAnswer.classList.add("is-error");
  } finally {
    stopTimer();
    setBusy(false);
  }
});

loadHealth();
