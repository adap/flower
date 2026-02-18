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

  function getCurrentVersion(versioningContainer) {
    if (!versioningContainer) {
      return "";
    }
    return versioningContainer.dataset.currentVersion || "";
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

  function getLatestStableVersion(versions) {
    for (const version of versions) {
      if (version && version.name && version.name !== "main") {
        return version.name;
      }
    }
    return "";
  }

  function removeOldVersionBanner() {
    const existing = document.getElementById("flwr-old-version-banner");
    if (existing) {
      existing.remove();
    }
  }

  function renderOldVersionBanner(versioningContainer, versions) {
    removeOldVersionBanner();

    if (!versioningContainer || versions.length === 0) {
      return;
    }

    const currentVersion = getCurrentVersion(versioningContainer);
    const latestStableVersion = getLatestStableVersion(versions);

    if (
      !currentVersion ||
      !latestStableVersion ||
      currentVersion === "main" ||
      currentVersion === latestStableVersion
    ) {
      return;
    }

    const docsBaseUrl = getDocsBaseUrl(versioningContainer);
    const currentLanguage = versioningContainer.dataset.currentLanguage || "en";
    const pagename = versioningContainer.dataset.pagename || "index";

    const banner = document.createElement("aside");
    banner.id = "flwr-old-version-banner";
    banner.className = "flwr-old-version-banner";

    const message = document.createElement("span");
    message.className = "flwr-old-version-banner__text";
    message.innerHTML = `This is documentation for an old version (<strong>${currentVersion}</strong>).`;

    const action = document.createElement("button");
    action.type = "button";
    action.className = "flwr-old-version-banner__action";
    action.textContent = "Switch to stable version";
    action.addEventListener("click", () => {
      checkPageExistence(latestStableVersion, currentLanguage, pagename, docsBaseUrl);
    });

    const close = document.createElement("button");
    close.type = "button";
    close.className = "flwr-old-version-banner__close";
    close.setAttribute("aria-label", "Dismiss old version warning");
    close.textContent = "×";
    close.addEventListener("click", () => {
      banner.remove();
    });

    banner.appendChild(message);
    banner.appendChild(action);
    banner.appendChild(close);

    const pageRoot = document.querySelector(".page");
    const announcement = document.querySelector(".announcement");
    if (announcement && announcement.parentNode) {
      announcement.parentNode.insertBefore(banner, announcement.nextSibling);
    } else if (pageRoot && pageRoot.parentNode) {
      pageRoot.parentNode.insertBefore(banner, pageRoot);
    }
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
    renderOldVersionBanner(versioningContainer, runtimeVersions);
    renderAnnouncement(runtimeConfig.announcement);
  });
})();
