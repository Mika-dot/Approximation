namespace ApproximationLib;

public sealed record ModelMetrics(double Mse, double Rmse, double Mae, double R2)
{
    internal static ModelMetrics Calculate(IReadOnlyList<double> actual, IReadOnlyList<double> predicted)
    {
        if (actual.Count != predicted.Count || actual.Count == 0) throw new ArgumentException("Metric arrays must have the same non-zero length.");
        double mean = actual.Average();
        double squared = 0, absolute = 0, total = 0;
        for (int i = 0; i < actual.Count; i++)
        {
            double error = predicted[i] - actual[i];
            squared += error * error;
            absolute += Math.Abs(error);
            double centered = actual[i] - mean;
            total += centered * centered;
        }
        double mse = squared / actual.Count;
        double r2 = total <= 1e-20 ? (mse <= 1e-20 ? 1.0 : 0.0) : 1.0 - squared / total;
        return new ModelMetrics(mse, Math.Sqrt(mse), absolute / actual.Count, r2);
    }
}
