using ApproximationLib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace SymbolicRegression
{
    class Program
    {
        static void Main(string[] args)
        {

            double[] X_DATA = { -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 };
            double[] Y_DATA = { 16, 9, 4, 1, 0, 1, 4, 9, 16, 25 }; // y = (x-1)^2

            // Простой режим
            var regressor = new SymbolicRegressor();
            var result = regressor.Fit(X_DATA, Y_DATA);
            Console.WriteLine($"Лучшая формула: {result.BestFormula}");
            Console.WriteLine($"MSE: {result.MSE}");

            // Автоподбор настроек
            var autoResult = regressor.AutoFit(X_DATA, Y_DATA);

            // Режим с отладкой и HTML отчетом
            var regressorWithReport = new SymbolicRegressor(new Config());
            var resultWithReport = regressorWithReport.FitWithReport(X_DATA, Y_DATA);

            
        }
    }
}