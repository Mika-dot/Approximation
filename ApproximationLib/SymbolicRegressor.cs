using ApproximationLib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace ApproximationLib
{
    public class SymbolicRegressor
    {

        // ==================== ПРИВАТНЫЕ ПОЛЯ ====================
        private readonly Config _config;
        private readonly Random _rng = new Random();
        private List<GenerationData> _generationHistory = new List<GenerationData>();

        public SymbolicRegressor(Config config = null)
        {
            _config = config ?? new Config();
        }

        // ==================== ОСНОВНЫЕ МЕТОДЫ ====================

        /// <summary>
        /// Простой режим: получаем только лучшую функцию
        /// </summary>
        public Result Fit(double[] xData, double[] yData)
        {
            var population = InitializePopulation();
            var startTime = DateTime.Now;

            for (int gen = 0; gen < _config.MaxGenerations; gen++)
            {
                EvaluatePopulationFitness(population, xData, yData);
                population = population.OrderBy(p => p.Fitness).ToList();
                var best = population[0];

                if (_config.EnableLogging)
                {
                    Console.WriteLine($"Поколение {gen}: MSE = {best.Mse:F6}, Формула = {best}");
                }

                if (best.Mse < 1e-6 || gen == _config.MaxGenerations - 1)
                {
                    var bestTree = population[0];
                    return new Result
                    {
                        BestFormula = bestTree.ToString(),
                        Fitness = bestTree.Fitness,
                        MSE = bestTree.Mse,
                        Size = bestTree.Size,
                        Generations = gen + 1,
                        TrainingTime = DateTime.Now - startTime,
                        Function = x => bestTree.Evaluate(x)
                    };
                }

                population = CreateNewPopulation(population);
            }

            return null;
        }

        /// <summary>
        /// Режим с отладкой: генерируем HTML отчет
        /// </summary>
        public Result FitWithReport(double[] xData, double[] yData)
        {
            _config.EnableLogging = true;
            _generationHistory.Clear();

            var population = InitializePopulation();
            var startTime = DateTime.Now;

            for (int gen = 0; gen < _config.MaxGenerations; gen++)
            {
                EvaluatePopulationFitness(population, xData, yData);
                population = population.OrderBy(p => p.Fitness).ToList();

                // Сохраняем данные поколения для отчета
                var generationData = new GenerationData(gen, population.Take(_config.TopSolutionsToTrack).ToList());
                _generationHistory.Add(generationData);

                var best = population[0];
                Console.WriteLine($"=== Поколение {gen} ===");
                Console.WriteLine($"Лучшая формула: {best}");
                Console.WriteLine($"Фитнес = MSE + λ*Size = {best.Mse:F6} + {_config.ParsimonyCoefficient}*{best.Size} = {best.Fitness:F6}");
                Console.WriteLine();

                if (best.Mse < 1e-6 || gen == _config.MaxGenerations - 1)
                {
                    GenerateHtmlReport(xData, yData);
                    var bestTree = population[0];
                    return new Result
                    {
                        BestFormula = bestTree.ToString(),
                        Fitness = bestTree.Fitness,
                        MSE = bestTree.Mse,
                        Size = bestTree.Size,
                        Generations = gen + 1,
                        TrainingTime = DateTime.Now - startTime,
                        Function = x => bestTree.Evaluate(x)
                    };
                }

                population = CreateNewPopulation(population);
            }

            return null;
        }

        /// <summary>
        /// Автоматический подбор настроек на основе данных
        /// </summary>
        public Result AutoFit(double[] xData, double[] yData)
        {
            var autoConfig = AutoTuneConfig(xData, yData);
            var regressor = new SymbolicRegressor(autoConfig);
            return regressor.Fit(xData, yData);
        }

        // ==================== АВТОПОДБОР НАСТРОЕК ====================

        private Config AutoTuneConfig(double[] xData, double[] yData)
        {
            var config = new Config();

            // Анализ сложности данных
            double rangeX = xData.Max() - xData.Min();
            double rangeY = yData.Max() - yData.Min();
            int dataSize = xData.Length;

            // Определяем сложность задачи
            bool isSimple = rangeY < 10 && dataSize < 20;
            bool isComplex = rangeY > 100 || dataSize > 50;

            if (isSimple)
            {
                config.PopulationSize = 1000;
                config.MaxGenerations = 200;
                config.MaxDepth = 3;
                config.Functions = new[] { "+", "-", "*", "protectedDiv", "sqrt", "abs" };
            }
            else if (isComplex)
            {
                config.PopulationSize = 3000;
                config.MaxGenerations = 800;
                config.MaxDepth = 6;
                config.ParsimonyCoefficient = 0.001;
            }
            else
            {
                config.PopulationSize = 2000;
                config.MaxGenerations = 500;
                config.MaxDepth = 4;
            }

            // Настройка на основе дисперсии данных
            double meanY = yData.Average();
            double variance = yData.Average(y => Math.Pow(y - meanY, 2));

            if (variance > 1000)
            {
                config.MutationRate = 0.95;
                config.CrossoverRate = 0.85;
            }

            Console.WriteLine($"Автонастройка: PopulationSize={config.PopulationSize}, " +
                            $"MaxGenerations={config.MaxGenerations}, MaxDepth={config.MaxDepth}");

            return config;
        }

        // ==================== ОСНОВНЫЕ АЛГОРИТМЫ ====================

        private List<ExpressionTree> InitializePopulation()
        {
            var pop = new List<ExpressionTree>();
            for (int i = 0; i < _config.PopulationSize; i++)
            {
                pop.Add(GrowTree(_rng.Next(1, _config.MaxDepth + 1)));
            }
            return pop;
        }

        private ExpressionTree GrowTree(int maxDepth)
        {
            return BuildTree(maxDepth, true);
        }

        private ExpressionTree BuildTree(int depth, bool isRoot = false)
        {
            if (depth == 0 || (depth > 1 && _rng.NextDouble() < 0.3))
            {
                if (_rng.NextDouble() < 0.7)
                    return new ExpressionTree("x");
                else
                    return new ExpressionTree(_rng.NextDouble() * 20 - 10);
            }
            else
            {
                string func = _config.Functions[_rng.Next(_config.Functions.Length)];
                if (IsUnary(func))
                {
                    return new ExpressionTree(func, BuildTree(depth - 1));
                }
                else
                {
                    return new ExpressionTree(func, BuildTree(depth - 1), BuildTree(depth - 1));
                }
            }
        }

        private bool IsUnary(string func)
        {
            return new[] { "sin", "cos", "tan", "asin", "acos", "atan",
                           "sinh", "cosh", "tanh", "log", "log10", "sqrt", "abs", "exp" }.Contains(func);
        }

        private void EvaluatePopulationFitness(List<ExpressionTree> population, double[] xData, double[] yData)
        {
            foreach (var individual in population)
            {
                individual.Fitness = EvaluateFitness(individual, xData, yData);
            }
        }

        private double EvaluateFitness(ExpressionTree tree, double[] xData, double[] yData)
        {
            double mse = 0;
            bool valid = true;

            for (int i = 0; i < xData.Length; i++)
            {
                try
                {
                    double pred = tree.Evaluate(xData[i]);
                    if (double.IsNaN(pred) || double.IsInfinity(pred))
                    {
                        valid = false;
                        break;
                    }
                    mse += Math.Pow(pred - yData[i], 2);
                }
                catch
                {
                    valid = false;
                    break;
                }
            }

            if (!valid)
            {
                tree.Mse = double.MaxValue;
                tree.Size = tree.GetSize();
                return double.MaxValue;
            }

            mse /= xData.Length;
            int size = tree.GetSize();
            tree.Mse = mse;
            tree.Size = size;
            return mse + _config.ParsimonyCoefficient * size;
        }

        private List<ExpressionTree> CreateNewPopulation(List<ExpressionTree> population)
        {
            var newPopulation = new List<ExpressionTree>();

            // Элитизм
            int survivorsCount = (int)(_config.PopulationSize * _config.SurvivalRate);
            for (int i = 0; i < survivorsCount; i++)
            {
                newPopulation.Add(Clone(population[i]));
            }

            // Случайные особи
            int randomCount = (int)(_config.PopulationSize * 0.02);
            for (int i = 0; i < randomCount && newPopulation.Count < _config.PopulationSize; i++)
            {
                newPopulation.Add(GrowTree(_rng.Next(1, _config.MaxDepth + 1)));
            }

            // Скрещивание и мутация
            while (newPopulation.Count < _config.PopulationSize)
            {
                ExpressionTree parent1 = TournamentSelection(population);
                ExpressionTree parent2 = TournamentSelection(population);

                ExpressionTree child1, child2;
                if (_rng.NextDouble() < _config.CrossoverRate)
                {
                    (child1, child2) = Crossover(parent1, parent2);
                }
                else
                {
                    child1 = Clone(parent1);
                    child2 = Clone(parent2);
                }

                if (_rng.NextDouble() < _config.MutationRate) Mutate(child1);
                if (_rng.NextDouble() < _config.MutationRate) Mutate(child2);

                if (child1.GetDepth() <= _config.MaxDepth) newPopulation.Add(child1);
                if (newPopulation.Count < _config.PopulationSize && child2.GetDepth() <= _config.MaxDepth)
                    newPopulation.Add(child2);
            }

            return newPopulation;
        }

        private ExpressionTree TournamentSelection(List<ExpressionTree> population)
        {
            var candidates = population.OrderBy(_ => _rng.Next()).Take(_config.TournamentSize).ToList();
            return candidates.OrderBy(p => p.Fitness).First();
        }

        private (ExpressionTree, ExpressionTree) Crossover(ExpressionTree parent1, ExpressionTree parent2)
        {
            var clone1 = Clone(parent1);
            var clone2 = Clone(parent2);

            var nodes1 = GetAllNodes(clone1).Where(n => !n.IsLeaf).ToList();
            var nodes2 = GetAllNodes(clone2).Where(n => !n.IsLeaf).ToList();

            if (nodes1.Count == 0 || nodes2.Count == 0)
                return (clone1, clone2);

            var node1 = nodes1[_rng.Next(nodes1.Count)];
            var node2 = nodes2[_rng.Next(nodes2.Count)];

            var subtree1 = CloneSubtree(node1);
            var subtree2 = CloneSubtree(node2);

            ReplaceSubtree(node1, subtree2);
            ReplaceSubtree(node2, subtree1);

            return (clone1, clone2);
        }

        private void Mutate(ExpressionTree tree)
        {
            var nodes = GetAllNodes(tree).ToList();
            if (nodes.Count == 0) return;

            var target = nodes[_rng.Next(nodes.Count)];
            double r = _rng.NextDouble();

            if (target.IsLeaf)
            {
                if (r < 0.5)
                    target.Value = "x";
                else
                    target.Value = (_rng.NextDouble() * 20 - 10).ToString("F4");
            }
            else
            {
                if (r < 0.7)
                {
                    var newSub = GrowTree(_config.MaxMutationDepth);
                    target.Value = newSub.Value;
                    target.Left = newSub.Left;
                    target.Right = newSub.Right;
                }
                else
                {
                    string old = target.Value;
                    string newFunc;
                    int attempts = 0;
                    do
                    {
                        newFunc = _config.Functions[_rng.Next(_config.Functions.Length)];
                        attempts++;
                        if (attempts > 10) break;
                    } while (IsUnary(old) != IsUnary(newFunc));

                    if (IsUnary(old) == IsUnary(newFunc))
                    {
                        target.Value = newFunc;
                    }
                }
            }
        }

        // ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

        private ExpressionTree CloneSubtree(ExpressionTree node)
        {
            return Clone(node);
        }

        private void ReplaceSubtree(ExpressionTree target, ExpressionTree newSub)
        {
            target.Value = newSub.Value;
            target.Left = Clone(newSub.Left);
            target.Right = Clone(newSub.Right);
        }

        private List<ExpressionTree> GetAllNodes(ExpressionTree node)
        {
            var list = new List<ExpressionTree> { node };
            if (node.Left != null) list.AddRange(GetAllNodes(node.Left));
            if (node.Right != null) list.AddRange(GetAllNodes(node.Right));
            return list;
        }

        public ExpressionTree Clone(ExpressionTree node)
        {
            if (node == null) return null;
            return new ExpressionTree(
                node.Value,
                Clone(node.Left),
                Clone(node.Right)
            );
        }

        // ==================== ГЕНЕРАЦИЯ ОТЧЕТА ====================

        private void GenerateHtmlReport(double[] xData, double[] yData)
        {
            var html = new StringBuilder();

            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html lang='ru'>");
            html.AppendLine("<head>");
            html.AppendLine("    <meta charset='UTF-8'>");
            html.AppendLine("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>");
            html.AppendLine("    <title>Symbolic Regression Report</title>");
            html.AppendLine("    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>");
            html.AppendLine("    <style>");
            html.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }");
            html.AppendLine("        .generation { background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }");
            html.AppendLine("        .generation h2 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }");
            html.AppendLine("        .solutions { display: flex; flex-wrap: wrap; gap: 20px; }");
            html.AppendLine("        .solution { flex: 1; min-width: 300px; background: #f9f9f9; padding: 15px; border-radius: 5px; }");
            html.AppendLine("        .solution h3 { margin-top: 0; color: #007acc; }");
            html.AppendLine("        .chart-container { position: relative; height: 300px; margin-top: 15px; }");
            html.AppendLine("        .stats { background: #e8f4fc; padding: 10px; border-radius: 5px; margin: 10px 0; }");
            html.AppendLine("    </style>");
            html.AppendLine("</head>");
            html.AppendLine("<body>");
            html.AppendLine("    <h1>📊 Отчет по символической регрессии</h1>");
            html.AppendLine($"    <div class='stats'>");
            html.AppendLine($"        <strong>Данные:</strong> x = [{string.Join(", ", xData)}], y = [{string.Join(", ", yData)}]<br>");
            html.AppendLine($"        <strong>Популяция:</strong> {_config.PopulationSize}, <strong>Поколений:</strong> {_generationHistory.Count}<br>");
            html.AppendLine($"        <strong>Лучший MSE:</strong> {_generationHistory.Last().Solutions[0].Mse:F6}");
            html.AppendLine("    </div>");

            foreach (var genData in _generationHistory)
            {
                html.AppendLine($"    <div class='generation' id='gen-{genData.GenerationNumber}'>");
                html.AppendLine($"        <h2>🎯 Поколение {genData.GenerationNumber}</h2>");
                html.AppendLine($"        <div class='stats'>");
                html.AppendLine($"            Лучший фитнес: {genData.Solutions[0].Fitness:F6}, Лучший MSE: {genData.Solutions[0].Mse:F6}");
                html.AppendLine("        </div>");
                html.AppendLine("        <div class='solutions'>");

                for (int i = 0; i < genData.Solutions.Count; i++)
                {
                    var solution = genData.Solutions[i];
                    html.AppendLine($"            <div class='solution'>");
                    html.AppendLine($"                <h3>#{i + 1}: {EscapeHtml(solution.ToString())}</h3>");
                    html.AppendLine($"                <div class='stats'>");
                    html.AppendLine($"                    <strong>MSE:</strong> {solution.Mse:F6}<br>");
                    html.AppendLine($"                    <strong>Размер:</strong> {solution.Size}<br>");
                    html.AppendLine($"                    <strong>Фитнес:</strong> {solution.Fitness:F6}");
                    html.AppendLine("                </div>");
                    html.AppendLine($"                <div class='chart-container'>");
                    html.AppendLine($"                    <canvas id='chart-gen{genData.GenerationNumber}-sol{i}'></canvas>");
                    html.AppendLine("                </div>");
                    html.AppendLine("            </div>");
                }

                html.AppendLine("        </div>");
                html.AppendLine("    </div>");
            }

            // JavaScript для графиков
            html.AppendLine("    <script>");
            html.AppendLine(GenerateJavaScriptHelpers());
            html.AppendLine("        function initializeCharts() {");
            html.AppendLine("            const colors = [");
            html.AppendLine("                'rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 205, 86)',");
            html.AppendLine("                'rgb(75, 192, 192)', 'rgb(153, 102, 255)', 'rgb(255, 159, 64)',");
            html.AppendLine("                'rgb(201, 203, 207)', 'rgb(255, 99, 255)', 'rgb(99, 255, 132)'");
            html.AppendLine("            ];");

            html.AppendLine("            const originalDataPoints = [");
            for (int i = 0; i < xData.Length; i++)
            {
                html.AppendLine($"                {{x: {xData[i]}, y: {yData[i]}}},");
            }
            html.AppendLine("            ];");

            foreach (var genData in _generationHistory)
            {
                for (int i = 0; i < genData.Solutions.Count; i++)
                {
                    var solution = genData.Solutions[i];
                    var formulaJs = EscapeJavaScript(solution.ToString());

                    html.AppendLine($"            const ctx_{genData.GenerationNumber}_{i} = document.getElementById('chart-gen{genData.GenerationNumber}-sol{i}').getContext('2d');");
                    html.AppendLine($"            const pred_{genData.GenerationNumber}_{i} = generatePredictionData('{formulaJs}');");
                    html.AppendLine($"            new Chart(ctx_{genData.GenerationNumber}_{i}, {{");
                    html.AppendLine("                type: 'scatter',");
                    html.AppendLine("                data: {");
                    html.AppendLine("                    datasets: [");
                    html.AppendLine("                        {");
                    html.AppendLine("                            label: 'Исходные данные',");
                    html.AppendLine("                            data: originalDataPoints,");
                    html.AppendLine("                            backgroundColor: 'rgb(0, 0, 0)',");
                    html.AppendLine("                            pointRadius: 6,");
                    html.AppendLine("                            pointStyle: 'circle'");
                    html.AppendLine("                        },");
                    html.AppendLine("                        {");
                    html.AppendLine($"                            label: 'Решение #{i + 1}',");
                    html.AppendLine($"                            data: pred_{genData.GenerationNumber}_{i},");
                    html.AppendLine($"                            borderColor: colors[{i % 9}],");
                    html.AppendLine("                            backgroundColor: colors[" + (i % 9) + "],");
                    html.AppendLine("                            pointRadius: 2,");
                    html.AppendLine("                            showLine: true,");
                    html.AppendLine("                            borderWidth: 1,");
                    html.AppendLine("                            fill: false");
                    html.AppendLine("                        }");
                    html.AppendLine("                    ]");
                    html.AppendLine("                },");
                    html.AppendLine("                options: {");
                    html.AppendLine("                    responsive: true,");
                    html.AppendLine("                    maintainAspectRatio: false,");
                    html.AppendLine("                    scales: {");
                    html.AppendLine("                        x: { title: { display: true, text: 'x' }, type: 'linear' },");
                    html.AppendLine("                        y: { title: { display: true, text: 'y' }, type: 'linear' }");
                    html.AppendLine("                    }");
                    html.AppendLine("                }");
                    html.AppendLine("            });");
                }
            }

            html.AppendLine("        }");
            html.AppendLine("        document.addEventListener('DOMContentLoaded', initializeCharts);");
            html.AppendLine("    </script>");
            html.AppendLine("</body>");
            html.AppendLine("</html>");

            File.WriteAllText(_config.OutputPath, html.ToString(), Encoding.UTF8);
        }

        private string GenerateJavaScriptHelpers()
        {
            return @"
        function add(a, b) { return a + b; }
        function sub(a, b) { return a - b; }
        function mul(a, b) { return a * b; }
        function div(a, b) { return Math.abs(b) < 1e-9 ? 1.0 : a / b; }
        function pow(a, b) {
            if (a < 0 && Math.abs(b - Math.round(b)) > 1e-9) return 0.0;
            const r = Math.pow(a, b);
            return isFinite(r) ? r : 0.0;
        }
        function sin(x) { return Math.sin(x); }
        function cos(x) { return Math.cos(x); }
        function tan(x) {
            const c = Math.cos(x);
            return Math.abs(c) < 1e-9 ? 0.0 : Math.sin(x) / c;
        }
        function asin(x) { return Math.asin(Math.max(-1, Math.min(1, x))); }
        function acos(x) { return Math.acos(Math.max(-1, Math.min(1, x))); }
        function atan(x) { return Math.atan(x); }
        function sinh(x) { return Math.sinh(x); }
        function cosh(x) { return Math.cosh(x); }
        function tanh(x) { return Math.tanh(x); }
        function log(x) { return x <= 0 ? -10.0 : Math.log(x); }
        function log10(x) { return x <= 0 ? -10.0 : Math.log10(x); }
        function sqrt(x) { return x < 0 ? 0.0 : Math.sqrt(x); }
        function abs(x) { return Math.abs(x); }
        function exp(x) { return Math.exp(x); }
        function min(a, b) { return Math.min(a, b); }
        function max(a, b) { return Math.max(a, b); }

        function evaluateExpression(expr, xVal) {
            let code = expr.replace(/x/g, '(' + xVal + ')');
            try {
                return eval(code);
            } catch (e) {
                return NaN;
            }
        }

        function generatePredictionData(formula) {
            const data = [];
            const step = 0.1;
            for (let x = -3; x <= 6; x += step) {
                const y = evaluateExpression(formula, x);
                if (isFinite(y)) {
                    data.push({x: x, y: y});
                }
            }
            return data;
        }
            ";
        }

        static string EscapeHtml(string text)
        {
            return text.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;").Replace("'", "&#39;");
        }

        static string EscapeJavaScript(string text)
        {
            return text.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
        }
    }
}