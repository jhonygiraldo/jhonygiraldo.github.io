(function () {
  "use strict";

  function initializePublicationSearch() {
    var input = document.querySelector("[data-publication-search]");
    var items = Array.prototype.slice.call(document.querySelectorAll("[data-publication-item]"));
    var count = document.querySelector("[data-publication-count]");

    if (!input || !items.length) return;

    function update() {
      var query = input.value.trim().toLocaleLowerCase();
      var visible = 0;

      items.forEach(function (item) {
        var matches = !query || item.textContent.toLocaleLowerCase().indexOf(query) !== -1;
        item.hidden = !matches;
        if (matches) visible += 1;
      });

      document.querySelectorAll("[data-publication-year]").forEach(function (heading) {
        var year = heading.getAttribute("data-publication-year");
        var yearHasResults = items.some(function (item) {
          return item.getAttribute("data-year") === year && !item.hidden;
        });
        heading.hidden = !yearHasResults;
      });

      if (count) {
        count.textContent = visible + (visible === 1 ? " publication" : " publications");
      }
    }

    input.addEventListener("input", update);
    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializePublicationSearch);
  } else {
    initializePublicationSearch();
  }
})();

