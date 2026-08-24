using System.Globalization;
using System.Net;
using System.Text;

namespace ApproximationLib;

internal static class ReportWriter
{
    public static void Write(string path, Result result, double[] actual, double[] predicted)
    {
        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        var html = new StringBuilder();
        html.AppendLine("<!doctype html><html lang='ru'><head><meta charset='utf-8'>");
        html.AppendLine("<meta name='viewport' content='width=device-width,initial-scale=1'>");
        html.AppendLine("<title>Approximation · Symbolic regression report</title>");
        html.AppendLine("<style>");
        html.AppendLine("*{box-sizing:border-box}body{margin:0;background:#07111f;color:#e9f1ff;font:15px/1.5 Inter,Segoe UI,sans-serif}main{max-width:1240px;margin:auto;padding:42px 24px 70px}.eyebrow{color:#55e6c1;text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:800}h1{font-size:clamp(34px,5vw,66px);line-height:1.02;margin:10px 0 16px;max-width:950px}.formula{font:600 20px/1.5 Consolas,monospace;color:#a9ffd9;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.card,.panel{background:linear-gradient(145deg,#101f35,#0b1729);border:1px solid #203858;border-radius:18px;box-shadow:0 16px 45px #0005}.card{padding:18px}.card span{display:block;color:#8fa4bf;font-size:12px;text-transform:uppercase;letter-spacing:.08em}.card b{font-size:25px}.panel{padding:22px;margin-top:16px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}h2{margin:0 0 14px;font-size:19px}.muted{color:#8fa4bf}.pareto{width:100%;border-collapse:collapse}.pareto td,.pareto th{padding:10px;border-bottom:1px solid #203858;text-align:left}.pareto th{color:#8fa4bf;font-size:12px;text-transform:uppercase}.pareto code{color:#a9ffd9;white-space:normal}svg{width:100%;height:auto;display:block}.axis{stroke:#49627f;stroke-width:1}.gridline{stroke:#203858;stroke-width:1}.line{fill:none;stroke:#55e6c1;stroke-width:3}.point{fill:#7da7ff;fill-opacity:.75}.ideal{stroke:#ffca6a;stroke-width:2;stroke-dasharray:7 6}.caption{fill:#8fa4bf;font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}}@media(max-width:480px){.grid{grid-template-columns:1fr}}</style></head><body><main>");
        html.AppendLine("<div class='eyebrow'>Approximation / scientific model discovery</div>");
        html.AppendLine("<h1>Отчёт символической регрессии</h1>");
        html.AppendLine($"<div class='formula'>{WebUtility.HtmlEncode(result.BestFormula)}</div>");
        html.AppendLine("<div class='grid'>");
        Metric(html, "R²", F(result.R2)); Metric(html, "RMSE", F(result.RMSE));
        Metric(html, "MAE", F(result.MAE)); Metric(html, "Узлов", result.Size.ToString(CultureInfo.InvariantCulture));
        Metric(html, "Поколений", result.Generations.ToString(CultureInfo.InvariantCulture));
        Metric(html, "Время", $"{result.TrainingTime.TotalSeconds:F2} с");
        Metric(html, "MSE", F(result.MSE)); Metric(html, "Validation loss", F(result.ValidationLoss));
        html.AppendLine("</div><div class='charts'>");
        html.AppendLine("<section class='panel'><h2>Наблюдения и прогноз</h2><p class='muted'>Чем ближе точки к диагонали, тем точнее модель.</p>");
        html.AppendLine(ScatterSvg(actual, predicted)); html.AppendLine("</section>");
        html.AppendLine("<section class='panel'><h2>Сходимость</h2><p class='muted'>Ошибка лучшей модели на обучении и валидации.</p>");
        html.AppendLine(HistorySvg(result.History)); html.AppendLine("</section></div>");
        html.AppendLine("<section class='panel'><h2>Парето-фронт: точность ↔ сложность</h2><table class='pareto'><thead><tr><th>Узлы</th><th>Train MSE</th><th>Validation</th><th>Формула</th></tr></thead><tbody>");
        foreach (ParetoSolution item in result.ParetoFront.Take(20))
            html.AppendLine($"<tr><td>{item.Size}</td><td>{F(item.TrainingLoss)}</td><td>{F(item.ValidationLoss)}</td><td><code>{WebUtility.HtmlEncode(item.Formula)}</code></td></tr>");
        html.AppendLine("</tbody></table></section>");
        html.AppendLine($"<p class='muted'>Признаки: {WebUtility.HtmlEncode(string.Join(", ", result.FeatureNames))}. Отчёт автономный: графики встроены в HTML и не требуют CDN.</p>");
        html.AppendLine("</main></body></html>");
        File.WriteAllText(path, html.ToString(), new UTF8Encoding(false));
    }

    private static void Metric(StringBuilder html, string label, string value) =>
        html.AppendLine($"<div class='card'><span>{WebUtility.HtmlEncode(label)}</span><b>{WebUtility.HtmlEncode(value)}</b></div>");

    private static string ScatterSvg(double[] actual, double[] predicted)
    {
        const int width = 560, height = 330, pad = 42;
        double min = Math.Min(actual.Min(), predicted.Min()), max = Math.Max(actual.Max(), predicted.Max());
        if (Math.Abs(max - min) < 1e-12) { min -= 1; max += 1; }
        double X(double v) => pad + (v - min) / (max - min) * (width - 2 * pad);
        double Y(double v) => height - pad - (v - min) / (max - min) * (height - 2 * pad);
        var svg = BeginSvg(width, height);
        Grid(svg, width, height, pad);
        svg.AppendLine($"<line class='ideal' x1='{X(min):F2}' y1='{Y(min):F2}' x2='{X(max):F2}' y2='{Y(max):F2}'/>");
        for (int i = 0; i < actual.Length; i++)
            svg.AppendLine($"<circle class='point' cx='{X(actual[i]):F2}' cy='{Y(predicted[i]):F2}' r='3.5'><title>y={F(actual[i])}; ŷ={F(predicted[i])}</title></circle>");
        svg.AppendLine($"<text class='caption' x='{width / 2}' y='{height - 7}' text-anchor='middle'>Наблюдаемое y</text>");
        svg.AppendLine("</svg>");
        return svg.ToString();
    }

    private static string HistorySvg(IReadOnlyList<GenerationData> history)
    {
        const int width = 560, height = 330, pad = 42;
        double[] losses = history.SelectMany(h => new[] { h.BestTrainingLoss, h.BestValidationLoss })
            .Where(v => double.IsFinite(v) && v >= 0).Select(v => Math.Log10(v + 1e-16)).ToArray();
        double min = losses.Length == 0 ? -12 : losses.Min(), max = losses.Length == 0 ? 0 : losses.Max();
        if (Math.Abs(max - min) < 1e-12) { min -= 1; max += 1; }
        double X(int i) => pad + i / (double)Math.Max(1, history.Count - 1) * (width - 2 * pad);
        double Y(double value) => height - pad - (Math.Log10(Math.Max(0, value) + 1e-16) - min) / (max - min) * (height - 2 * pad);
        var svg = BeginSvg(width, height); Grid(svg, width, height, pad);
        string train = string.Join(" ", history.Select((h, i) => $"{X(i):F2},{Y(h.BestTrainingLoss):F2}"));
        string validation = string.Join(" ", history.Select((h, i) => $"{X(i):F2},{Y(h.BestValidationLoss):F2}"));
        svg.AppendLine($"<polyline class='line' points='{train}'/>");
        svg.AppendLine($"<polyline points='{validation}' fill='none' stroke='#7da7ff' stroke-width='2' stroke-dasharray='6 5'/>");
        svg.AppendLine($"<text class='caption' x='{width / 2}' y='{height - 7}' text-anchor='middle'>Поколение · логарифмическая шкала ошибки</text>");
        svg.AppendLine("</svg>");
        return svg.ToString();
    }

    private static StringBuilder BeginSvg(int width, int height) =>
        new($"<svg viewBox='0 0 {width} {height}' role='img' xmlns='http://www.w3.org/2000/svg'>");

    private static void Grid(StringBuilder svg, int width, int height, int pad)
    {
        for (int i = 0; i <= 5; i++)
        {
            double x = pad + i / 5.0 * (width - 2 * pad), y = pad + i / 5.0 * (height - 2 * pad);
            svg.AppendLine($"<line class='gridline' x1='{x:F2}' y1='{pad}' x2='{x:F2}' y2='{height - pad}'/><line class='gridline' x1='{pad}' y1='{y:F2}' x2='{width - pad}' y2='{y:F2}'/>");
        }
        svg.AppendLine($"<line class='axis' x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}'/><line class='axis' x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}'/>");
    }

    private static string F(double value) => value.ToString("G6", CultureInfo.InvariantCulture);
}
