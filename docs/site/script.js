"use strict";

const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector("#site-nav");
menuButton.addEventListener("click", () => {
  const expanded = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!expanded));
  navigation.classList.toggle("is-open", !expanded);
});
navigation.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    menuButton.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
  });
});

// Each column repeats the same sequence so the background loops continuously.
const gallery = document.querySelector(".image-gallery");
const images = [
  { src: "assets/chest-xray.jpg", label: "Chest X-ray",
    dataset: "Images + labels", kind: "xray" },
  { src: "assets/fundus.jpg", label: "Fundus", kind: "fundus" },
  { src: "assets/oct.jpg", label: "OCT", kind: "oct" },
];
for (let columnIndex = 0; columnIndex < 7; columnIndex += 1) {
  const column = document.createElement("div");
  column.className = "gallery-column";
  for (let repeat = 0; repeat < 2; repeat += 1) {
    for (let imageIndex = 0; imageIndex < 4; imageIndex += 1) {
      const source = images[(imageIndex + columnIndex) % images.length];
      const card = document.createElement("div");
      card.className = `gallery-card ${source.kind}`;
      const image = document.createElement("img");
      image.src = source.src;
      image.alt = "";
      image.width = 180;
      image.height = 205;
      image.decoding = "async";
      const label = document.createElement("span");
      label.textContent = source.label;
      card.append(image, label);
      column.append(card);
    }
  }
  gallery.append(column);
}

const taskData = {
  classification: {
    metric: "AUC / accuracy",
    image: "assets/fundus.jpg",
    label: "Fundus photograph",
    dataset: "Images + labels",
    description: "Single-label classification supports accuracy, AUC, F1, macro F1, and quadratic kappa.",
  },
  multilabel: {
    metric: "Mean AUC",
    image: "assets/chest-xray.jpg",
    label: "Chest X-ray",
    dataset: "Images + labels",
    description: "Multilabel classification uses one binary label per finding and supports mean AUC, macro F1, and micro F1.",
  },
  segmentation: {
    metric: "Dice / IoU",
    image: "assets/chest-xray.jpg",
    label: "Segmentation input",
    dataset: "Images + masks",
    description: "Binary segmentation pairs each image with a foreground mask and supports Dice and IoU metrics.",
  },
};

function wireTabs(selector, onSelect) {
  const tabs = [...document.querySelectorAll(selector)];
  function selectTab(tab, focus = false) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    onSelect(tab);
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      selectTab(tabs[next], true);
    });
  });
}

wireTabs("[data-task]", (tab) => {
  const task = taskData[tab.dataset.task];
  document.querySelectorAll(".metric-label").forEach((label) => { label.textContent = task.metric; });
  document.querySelector(".input-label").textContent = task.label;
  document.getElementById("dataset-label").textContent = task.dataset;
  document.querySelector(".mobile-dataset-label").textContent = `${task.dataset}, fixed splits`;
  document.getElementById("flow-image").setAttribute("href", task.image);
  document.querySelector(".mobile-flow-image").src = task.image;
  document.getElementById("task-description").textContent = task.description;
  document.getElementById("task-diagram").setAttribute("aria-labelledby", tab.id);
  document.getElementById("task-recipe").href = `https://github.com/mattlungrenmd/autoresearch-medimage/blob/main/recipes/template_${tab.dataset.task}.yaml`;
});

wireTabs("[data-code]", (tab) => {
  document.querySelectorAll(".code-window [role=tabpanel]").forEach((panel) => {
    panel.hidden = panel.id !== tab.getAttribute("aria-controls");
  });
  document.getElementById("copy-status").textContent = "";
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    const table = button.dataset.view === "table";
    document.getElementById("results-chart").hidden = table;
    document.getElementById("results-table").hidden = !table;
    document.querySelectorAll("[data-view]").forEach((item) => {
      item.setAttribute("aria-pressed", String(item === button));
    });
  });
});

document.getElementById("copy-code").addEventListener("click", async () => {
  const code = document.querySelector('.code-window [role="tabpanel"]:not([hidden]) code');
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = "Commands copied to clipboard.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Commands selected. Use your browser’s copy command.";
  }
});

// Pause diagrams outside the viewport; the method animation never runs training.
const scenes = new IntersectionObserver((entries) => {
  entries.forEach((entry) => entry.target.classList.toggle("is-visible", entry.isIntersecting));
}, { threshold: 0.08 });
document.querySelectorAll(".animation-scene").forEach((scene) => scenes.observe(scene));

const stageDescriptions = {
  search: "Search: compare seed recipes with short runs under the same time budget.",
  refine: "Refine: change training choices, confirm borderline gains, and retain candidate lineage.",
  train: "Train: select diverse candidates and run longer follow-up experiments.",
  evaluate: "Evaluate: report held-out test metrics for the saved finalists’ checkpoints.",
};
const stages = Object.keys(stageDescriptions);
const searchVisual = document.querySelector(".search-visual");
let stageIndex = 0;
let lastStageChange = performance.now();
function selectStage(stage) {
  stageIndex = stages.indexOf(stage);
  searchVisual.dataset.stage = stage;
  document.getElementById("stage-description").textContent = stageDescriptions[stage];
  document.querySelectorAll("button[data-stage]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.stage === stage));
  });
  lastStageChange = performance.now();
}
document.querySelectorAll("button[data-stage]").forEach((button) => {
  button.addEventListener("click", () => selectStage(button.dataset.stage));
});

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const motionButton = document.getElementById("motion-toggle");
let motionPaused = reducedMotion.matches;
function updateMotion() {
  document.body.classList.toggle("motion-paused", motionPaused);
  document.body.classList.toggle("motion-enabled", !motionPaused);
  motionButton.setAttribute("aria-pressed", String(motionPaused));
  motionButton.querySelector(".motion-label").textContent = motionPaused ? "Play animations" : "Pause animations";
  motionButton.firstElementChild.textContent = motionPaused ? "▷" : "Ⅱ";
}
updateMotion();
motionButton.addEventListener("click", () => {
  motionPaused = !motionPaused;
  lastStageChange = performance.now();
  updateMotion();
});
reducedMotion.addEventListener("change", (event) => {
  motionPaused = event.matches;
  lastStageChange = performance.now();
  updateMotion();
});
setInterval(() => {
  if (motionPaused || document.hidden || !searchVisual.classList.contains("is-visible")) return;
  if (performance.now() - lastStageChange >= 6000) selectStage(stages[(stageIndex + 1) % stages.length]);
}, 500);

// Keep the connector paths attached to their HTML nodes at every desktop size.
const searchGraph = document.querySelector(".search-graph");
const searchWires = document.querySelector(".search-wires");
function updateSearchWires() {
  if (getComputedStyle(searchWires).display === "none") return;
  const bounds = searchGraph.getBoundingClientRect();
  searchWires.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
  const localBounds = (element) => {
    const rect = element.getBoundingClientRect();
    return { left: rect.left - bounds.left, right: rect.right - bounds.left, y: rect.top - bounds.top + rect.height / 2 };
  };
  const seed = localBounds(document.querySelector(".seed-node"));
  const roots = [...document.querySelectorAll(".candidate.root")].map(localBounds);
  const retained = [...document.querySelectorAll(".candidate.selected")].map(localBounds);
  const finalists = [...document.querySelectorAll(".finalist-node")].map(localBounds);
  const paths = roots.map((node) => {
    const bend = (seed.right + node.left) / 2;
    return `M${seed.right + 2} ${seed.y}H${bend}V${node.y}H${node.left - 2}`;
  });
  retained.forEach((node, index) => {
    const target = finalists[index];
    const bend = (node.right + target.left) / 2;
    paths.push(`M${node.right + 2} ${node.y}H${bend}V${target.y}H${target.left - 2}`);
  });
  const makePath = (d, index, animated) => {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    if (animated) {
      path.setAttribute("pathLength", "1");
      path.setAttribute("class", `signal delay-${index % 5}`);
    }
    return path;
  };
  searchWires.querySelector(".connections").replaceChildren(...paths.map((d, index) => makePath(d, index, false)));
  searchWires.querySelector(".search-signal-paths").replaceChildren(...paths.map((d, index) => makePath(d, index, true)));
}
new ResizeObserver(updateSearchWires).observe(searchGraph);
updateSearchWires();
