/*
  Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
  Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»
  Клиентские сценарии интерфейса: мобильное меню, подтверждение опасных
  действий, отслеживание фоновых задач без перезагрузки страницы.
*/
(function () {
  "use strict";

  // Открытие и закрытие бокового меню на узких экранах
  var toggle = document.getElementById("menu-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      document.getElementById("sidebar").classList.toggle("open");
    });
  }

  // Кнопки и формы с атрибутом data-confirm спрашивают подтверждение
  document.addEventListener("submit", function (event) {
    var message = event.target.getAttribute("data-confirm");
    if (message && !window.confirm(message)) {
      event.preventDefault();
    }
  });

  // Автоматическая отправка формы фильтра при выборе значения в списке
  document.querySelectorAll("select[data-autosubmit]").forEach(function (select) {
    select.addEventListener("change", function () { select.form.submit(); });
  });

  // Отслеживание фоновой задачи: блок с data-poll-url опрашивает сервер и
  // обновляет полосу прогресса; по завершении задачи страница перезагружается
  document.querySelectorAll("[data-poll-url]").forEach(function (box) {
    var url = box.getAttribute("data-poll-url");
    var bar = box.querySelector(".progress i");
    var text = box.querySelector(".poll-message");
    var timer = window.setInterval(function () {
      fetch(url, { headers: { "Accept": "application/json" } })
        .then(function (response) { return response.json(); })
        .then(function (state) {
          if (bar && typeof state.progress === "number") {
            bar.style.width = Math.round(state.progress * 100) + "%";
          }
          if (text && state.message) { text.textContent = state.message; }
          if (state.status !== "running" && state.status !== "new") {
            window.clearInterval(timer);
            window.location.reload();
          } else if (state.epochs && box.getAttribute("data-epochs") !== String(state.epochs)) {
            // Появились новые эпохи — перерисовываем кривые обучения
            box.setAttribute("data-epochs", String(state.epochs));
            var chart = document.getElementById("live-chart");
            if (chart && state.chart) { chart.innerHTML = state.chart; }
          }
        })
        .catch(function () { /* временная недоступность сервера — пробуем снова */ });
    }, 2000);
  });
})();
