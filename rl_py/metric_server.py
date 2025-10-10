import time
from typing import final, Any
from flask import Flask, render_template_string
import threading
import numpy as np


@final
class MetricServer:
    def __init__(self, port=5000):
        self.game_results: list[dict[str, Any]] = []
        self.training_metrics: list[dict[str, Any]] = []
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        self.start_server()

    def setup_routes(self):
        @self.app.route("/")
        def dashboard():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route("/data")
        def data():
            return {
                "game_results": self.game_results,
                "training_metrics": self.training_metrics,
                "summary": self.get_summary_stats(),
            }

    def start_server(self):
        """Start the Flask server in a background thread"""

        def run():
            self.app.run(
                host="0.0.0.0", port=self.port, debug=False, use_reloader=False
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        print(f"📊 Metrics dashboard available at: http://localhost:{self.port}")

    def record_result(
        self,
        opponent_id: str,
        player_id: int,
        score: list[int],
        won: bool,
        training_step: int,
    ):
        gd = score[0] - score[1] if player_id == 1 else score[1] - score[0]
        self.game_results.append(
            {
                "training_step": training_step,
                "opponent_id": opponent_id,
                "player_id": player_id,
                "won": won,
                "goal_difference": gd,
                "timestamp": time.time(),
            }
        )

    def record_metrics(self, metrics: dict[str, float], training_step: int):
        metrics["training_step"] = training_step
        metrics["timestamp"] = time.time()
        self.training_metrics.append(metrics)

    def get_summary_stats(self) -> dict:
        """Calculate current summary statistics"""
        if not self.game_results:
            return {}

        recent_games = self.game_results[-100:]  # Last 100 games

        stats = {
            "total_games": len(self.game_results),
            "recent_games": len(recent_games),
        }

        # Win rates by opponent and player
        win_rates = {}
        for opponent in set(g["opponent_id"] for g in recent_games):
            for player in [1, 2]:
                games = [
                    g
                    for g in recent_games
                    if g["opponent_id"] == opponent and g["player_id"] == player
                ]
                if games:
                    wins = sum(1 for g in games if g["won"])
                    win_rates[f"{opponent}_p{player}"] = {
                        "win_rate": wins / len(games),
                        "games": len(games),
                    }

        stats["win_rates"] = win_rates

        # Recent goal difference
        if recent_games:
            stats["avg_goal_difference"] = np.mean(
                [g["goal_difference"] for g in recent_games]
            )

        # Latest training metrics
        if self.training_metrics:
            stats["latest_metrics"] = self.training_metrics[-1]

        return stats


# HTML Template with Chart.js
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; }
        .refresh { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Training Dashboard</h1>

        <div class="summary">
            <h3>Summary Stats</h3>
            <div id="summary"></div>
        </div>

        <div class="refresh">
            <button onclick="loadData()">🔄 Refresh</button>
            Auto-refresh: <input type="checkbox" id="autoRefresh" checked>
            <span id="lastUpdate"></span>
        </div>

        <div class="charts">
            <div class="chart-container">
                <h3>Win Rate Over Time</h3>
                <canvas id="winRateChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Goal Difference</h3>
                <canvas id="goalDiffChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Training Loss</h3>
                <canvas id="lossChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Recent Performance</h3>
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let charts = {};
        let autoRefresh = true;

        // Initialize checkboxes
        document.getElementById('autoRefresh').addEventListener('change', function() {
            autoRefresh = this.checked;
        });

        function loadData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    updateSummary(data.summary);
                    updateCharts(data);
                    document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
                });
        }

        function updateSummary(summary) {
            const summaryDiv = document.getElementById('summary');
            let html = `<p>Total Games: ${summary.total_games || 0}</p>`;

            if (summary.win_rates) {
                html += '<h4>Recent Win Rates (last 100 games):</h4>';
                for (const [key, stats] of Object.entries(summary.win_rates)) {
                    html += `<p>${key}: ${(stats.win_rate * 100).toFixed(1)}% (${stats.games} games)</p>`;
                }
            }

            if (summary.avg_goal_difference) {
                html += `<p>Avg Goal Difference: ${summary.avg_goal_difference.toFixed(2)}</p>`;
            }

            summaryDiv.innerHTML = html;
        }

        function updateCharts(data) {
            updateWinRateChart(data.game_results);
            updateGoalDiffChart(data.game_results);
            updateLossChart(data.training_metrics);
            updatePerformanceChart(data.game_results);
        }

        function updateWinRateChart(gameResults) {
            const ctx = document.getElementById('winRateChart').getContext('2d');

            // Group by opponent and player, then calculate rolling win rate
            const opponents = [...new Set(gameResults.map(g => g.opponent_id))];
            const datasets = [];

            opponents.forEach(opponent => {
                [1, 2].forEach(player => {
                    const playerGames = gameResults
                        .filter(g => g.opponent_id === opponent && g.player_id === player)
                        .sort((a, b) => parseInt(a.training_step) - parseInt(b.training_step));

                    if (playerGames.length > 0) {
                        const steps = [];
                        const winRates = [];
                        let windowSize = 50;

                        for (let i = 1; i <= playerGames.length; i++) {
                            const start = Math.max(0, i - windowSize);
                            const samples = i - start;
                            const window = playerGames.slice(start, i);
                            const wins = window.filter(g => g.won).length;
                            steps.push(window[window.length-1].training_step);
                            winRates.push((wins / samples) * 100);
                        }


                        datasets.push({
                            label: `${opponent} P${player}`,
                            data: winRates.map((rate, idx) => ({x: parseInt(steps[idx]), y: rate})),
                            borderWidth: 2,
                            tension: 0.1
                        });
                    }
                });
            });

            if (charts.winRateChart) {
                charts.winRateChart.data.datasets = datasets;
                charts.winRateChart.update('none');
            } else {
                charts.winRateChart = new Chart(ctx, {
                    type: 'line',
                    data: { datasets },
                    options: {
                        scales: {
                            y: {
                                title: { display: true, text: 'Win Rate %' },
                                min: 0, max: 100
                            },
                            x: {
                                title: { display: true, text: 'Training Step' },
                                type: 'linear', min: 0
                            }
                        }
                    }
                });
            }
        }

        function updateGoalDiffChart(gameResults) {
            const ctx = document.getElementById('goalDiffChart').getContext('2d');

            // Calculate rolling average goal difference
            const steps = [];
            const goalDiffs = [];
            const windowSize = 50;

            for (let i = 1; i <= gameResults.length; i++) {
                const start = Math.max(0, i - windowSize);
                const window = gameResults.slice(start, i);
                const samples = i - start;
                const avgDiff = window.reduce((sum, g) => sum + g.goal_difference, 0) / samples;
                steps.push(parseInt(window[window.length-1].training_step));
                goalDiffs.push(avgDiff);
            }

            if (charts.goalDiffChart) {
                charts.goalDiffChart.data.datasets[0].data = goalDiffs.map((diff, idx) => ({x: steps[idx], y: diff}));
                charts.goalDiffChart.update('none');
            } else {
                charts.goalDiffChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [{
                            label: 'Avg Goal Difference',
                            data: goalDiffs.map((diff, idx) => ({x: steps[idx], y: diff})),
                            borderColor: 'rgb(75, 192, 192)',
                            borderWidth: 2,
                            tension: 0.1
                        }]
                    },
                    options: {
                        scales: {
                            y: { title: { display: true, text: 'Goal Difference' } },
                            x: { title: { display: true, text: 'Training Step' }, type: 'linear', min: 0 }
                        }
                    }
                });
            }
        }

        function updateLossChart(trainingMetrics) {
            const ctx = document.getElementById('lossChart').getContext('2d');

            if (trainingMetrics.length === 0) return;

            const lossData = trainingMetrics
                .filter(m => m.loss !== undefined)
                .map(m => ({x: m.training_step, y: m.loss}));

            if (charts.lossChart) {
                charts.lossChart.data.datasets[0].data = lossData;
                charts.lossChart.update('none');
            } else {
                charts.lossChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [{
                            label: 'Training Loss',
                            data: lossData,
                            borderColor: 'rgb(255, 99, 132)',
                            borderWidth: 1,
                            pointRadius: 0
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                title: { display: true, text: 'Loss' },
                                type: 'logarithmic'
                            },
                            x: {
                                title: { display: true, text: 'Training Step' },
                                type: 'linear'
                            }
                        }
                    }
                });
            }
        }

        function updatePerformanceChart(gameResults) {
            const ctx = document.getElementById('performanceChart').getContext('2d');

            // Last 20 games performance
            const recentGames = gameResults.slice(-20);
            const labels = recentGames.map((g, i) => i + 1);
            const goalDiffs = recentGames.map(g => g.goal_difference);
            const won = recentGames.map(g => g.won ? 1 : -1);

            if (charts.performanceChart) {
                charts.performanceChart.data.labels = labels;
                charts.performanceChart.data.datasets[0].data = goalDiffs;
                charts.performanceChart.data.datasets[1].data = won;
                charts.performanceChart.update('none');
            } else {
                charts.performanceChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Goal Difference',
                                data: goalDiffs,
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                yAxisID: 'y'
                            },
                            {
                                label: 'Win/Loss',
                                data: won,
                                type: 'line',
                                borderColor: 'rgb(255, 99, 132)',
                                borderWidth: 2,
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: { display: true, text: 'Goal Difference' }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                min: -1.5,
                                max: 1.5,
                                ticks: {
                                    callback: function(value) {
                                        return value === 1 ? 'Win' : value === -1 ? 'Loss' : '';
                                    }
                                },
                                grid: {
                                    drawOnChartArea: false
                                }
                            }
                        }
                    }
                });
            }
        }

        // Auto-refresh every 5 seconds
        setInterval(() => {
            if (autoRefresh) {
                loadData();
            }
        }, 5000);

        // Initial load
        loadData();
    </script>
</body>
</html>
"""

# Usage example:
if __name__ == "__main__":
    # Test the dashboard
    metrics = MetricServer(port=5001)

    # Simulate some data
    import random

    for step in range(1000):
        score = [random.randint(0, 10), random.randint(0, 10)]
        player_id = 1 if random.random() > 0.5 else 2
        won = score[0] > score[1] if player_id == 1 else score[1] > score[0]
        metrics.record_result(
            opponent_id="heuristic",
            player_id=player_id,
            score=score,
            won=won,
            training_step=step,
        )

        if step % 10 == 0:
            metrics.record_metrics({"loss": random.expovariate(1.0)}, step)

    print("Test data generated. Dashboard should be running...")
    input("Press Enter to exit...")
