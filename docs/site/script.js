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

function wireTabs(selector, panelId) {
  const tabs = [...document.querySelectorAll(selector)];
  function selectTab(tab, focus = false) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
    });
    if (focus) tab.focus();
    if (panelId) document.getElementById(panelId).textContent = "";
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      selectTab(tabs[next], true);
    });
  });
}
wireTabs("[data-task]");
wireTabs("[data-code]", "copy-status");

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
  const panel = document.querySelector('.code-window [role="tabpanel"]:not([hidden])');
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(panel.querySelector("code").textContent);
    status.textContent = "Commands copied to clipboard.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(panel.querySelector("code"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Commands selected. Use your browser’s copy command.";
  }
});
