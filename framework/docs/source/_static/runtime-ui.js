(function () {
  "use strict";

  function normalizeVersions(rawVersions) {
    if (!Array.isArray(rawVersions)) {
      return [];
    }
    return rawVersions.filter(
      (item) => item && typeof item.name === "string" && item.name.trim() !== "",
    );
  }

  function getVersioningContainer() {
    return document.querySelector("[data-flwr-versioning]");
  }

  function getDocsBaseUrl(versioningContainer) {
    if (!versioningContainer) {
      return "https://flower.ai/docs/framework";
    }
    return (
      versioningContainer.dataset.docsBaseUrl || "https://flower.ai/docs/framework"
    );
  }

  function getRuntimeConfigUrl(versioningContainer) {
    if (!versioningContainer) {
      return "https://flower.ai/docs/framework-config/runtime-ui.json";
    }
    return (
      versioningContainer.dataset.runtimeConfigUrl ||
      "https://flower.ai/docs/framework-config/runtime-ui.json"
    );
  }

  async function checkPageExistence(versionName, currentLanguage, pagename, docsBaseUrl) {
    const newUrl = `${docsBaseUrl}/${versionName}/${currentLanguage}/${pagename}.html`;
    const fallbackUrl = `${docsBaseUrl}/${versionName}/${currentLanguage}/index.html#references`;

    try {
      const response = await fetch(newUrl);
      window.location.href = response.ok ? newUrl : fallbackUrl;
    } catch (error) {
      console.error("Error:", error);
      window.location.href = fallbackUrl;
    }
  }

  function bindVersionLinks(versioningContainer) {
    if (!versioningContainer) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    const links = versioningContainer.querySelectorAll("a[data-version-name]");
    links.forEach((link) => {
      if (link.dataset.boundVersionClick === "true") {
        return;
      }
      const versionName = link.dataset.versionName;
      if (!versionName) {
        return;
      }
      link.addEventListener("click", (event) => {
        event.preventDefault();
        checkPageExistence(versionName, currentLanguage, pagename, docsBaseUrl);
      });
      link.dataset.boundVersionClick = "true";
    });
  }

  function renderVersionList(versioningContainer, versions) {
    if (!versioningContainer || versions.length === 0) {
      return;
    }

    const list = versioningContainer.querySelector("[data-flwr-version-list]");
    if (!list) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    list.innerHTML = "";

    versions.forEach((item) => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = `${docsBaseUrl}/${item.name}/${currentLanguage}/${pagename}.html`;
      link.dataset.versionName = item.name;
      link.textContent = item.name;
      li.appendChild(link);
      list.appendChild(li);
    });

    bindVersionLinks(versioningContainer);
  }

  function renderAnnouncement(runtimeAnnouncement) {
    const announcementContent = document.getElementById("flwr-runtime-announcement");
    if (!announcementContent) {
      return;
    }

    const announcementContainer = announcementContent.closest(".announcement");
    if (!announcementContainer) {
      return;
    }

    const hasAnnouncement =
      runtimeAnnouncement &&
      runtimeAnnouncement.enabled === true &&
      typeof runtimeAnnouncement.html === "string" &&
      runtimeAnnouncement.html.trim() !== "";

    if (!hasAnnouncement) {
      announcementContent.innerHTML = "";
      announcementContainer.style.display = "none";
      return;
    }

    announcementContent.innerHTML = runtimeAnnouncement.html;
    announcementContainer.style.removeProperty("display");
  }

  function hideEmptyAnnouncementFallback() {
    const announcementContent = document.getElementById("flwr-runtime-announcement");
    if (!announcementContent) {
      return;
    }

    const announcementContainer = announcementContent.closest(".announcement");
    if (!announcementContainer) {
      return;
    }

    if (announcementContent.textContent.trim() === "") {
      announcementContainer.style.display = "none";
    }
  }

  async function loadRuntimeConfig(runtimeConfigUrl) {
    try {
      const response = await fetch(runtimeConfigUrl, { cache: "no-store" });
      if (!response.ok) {
        return null;
      }
      return await response.json();
    } catch (error) {
      console.warn("Could not load docs runtime metadata:", error);
      return null;
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const versioningContainer = getVersioningContainer();
    bindVersionLinks(versioningContainer);
    hideEmptyAnnouncementFallback();

    const runtimeConfigUrl = getRuntimeConfigUrl(versioningContainer);
    const runtimeConfig = await loadRuntimeConfig(runtimeConfigUrl);
    if (!runtimeConfig || typeof runtimeConfig !== "object") {
      return;
    }

    const runtimeVersions = normalizeVersions(runtimeConfig.versions);
    renderVersionList(versioningContainer, runtimeVersions);
    renderAnnouncement(runtimeConfig.announcement);
  });
})();
