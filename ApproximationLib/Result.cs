using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApproximationLib
{
    // ==================== РЕЗУЛЬТАТ ОБУЧЕНИЯ ====================
    public class Result
    {
        public string BestFormula { get; set; }
        public double Fitness { get; set; }
        public double MSE { get; set; }
        public int Size { get; set; }
        public int Generations { get; set; }
        public TimeSpan TrainingTime { get; set; }
        public Func<double, double> Function { get; set; }
    }
}
