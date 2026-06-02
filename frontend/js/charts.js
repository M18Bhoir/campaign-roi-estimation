/**
 * Chart rendering helpers using Chart.js.
 * Each function takes a canvas ID and data, returns a Chart instance.
 */

export function renderROIGauge(canvasId, roiValue) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    const clampedROI = Math.min(Math.max(roiValue, -100), 500);
    const pct = ((clampedROI + 100) / 600) * 100;

    return new Chart(ctx, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [pct, 100 - pct],
                backgroundColor: [
                    roiValue >= 100 ? "#22c55e" : roiValue >= 0 ? "#f59e0b" : "#ef4444",
                    "#1e293b"
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: "75%",
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            rotation: -90,
            circumference: 180
        }
    });
}

export function renderFeatureImportanceChart(canvasId, importances) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    const labels = Object.keys(importances).map(k => k.replace(/_/g, " "));
    const values = Object.values(importances);

    return new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Feature Importance",
                data: values.map(v => +(v * 100).toFixed(1)),
                backgroundColor: "#6366f1",
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: "Importance (%)" } }
            }
        }
    });
}

export function renderConfidenceInterval(canvasId, lower, predicted, upper, label) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
        type: "bar",
        data: {
            labels: [label],
            datasets: [
                { label: "Lower (95% CI)", data: [lower], backgroundColor: "#fbbf24" },
                { label: "Predicted ROI", data: [predicted], backgroundColor: "#6366f1" },
                { label: "Upper (95% CI)", data: [upper], backgroundColor: "#34d399" }
            ]
        },
        options: {
            plugins: { legend: { position: "bottom" } },
            scales: { y: { title: { display: true, text: "ROI %" } } }
        }
    });
}
