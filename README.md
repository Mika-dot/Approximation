# Symbolic Regression Library

Библиотека для автоматического поиска математических формул по данным с помощью генетического программирования.

## 🚀 Возможности

- **Автоматический поиск формул** - находит математические выражения, описывающие ваши данные
- **Три режима работы** - простой, с отладкой и автоподбором параметров
- **HTML отчеты** - визуализация процесса обучения с графиками
- **Гибкая настройка** - полный контроль над параметрами алгоритма
- **Автоподбор параметров** - автоматическая настройка под сложность данных

## 📦 Установка

Добавьте библиотеку в ваш C# проект.

## 🛠 Быстрый старт

```csharp
// Ваши данные
double[] xData = { -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 };
double[] yData = { 16, 9, 4, 1, 0, 1, 4, 9, 16, 25 }; // y = (x-1)²

// Простой режим
var regressor = new SymbolicRegressor();
var result = regressor.Fit(xData, yData);

Console.WriteLine($"Формула: {result.BestFormula}");
Console.WriteLine($"MSE: {result.MSE:F6}");

// Использование найденной функции
double prediction = result.Function(2.5);
```

## 📊 Режимы работы

### 1. Простой режим
```csharp
var result = regressor.Fit(xData, yData);
```

### 2. Режим с отчетом
```csharp
var config = new SymbolicRegressor.Config 
{
    EnableLogging = true,
    OutputPath = "report.html"
};
var regressor = new SymbolicRegressor(config);
var result = regressor.FitWithReport(xData, yData);
```

### 3. Автоматический режим
```csharp
var result = regressor.AutoFit(xData, yData);
```

## ⚙️ Настройка параметров

```csharp
var config = new SymbolicRegressor.Config
{
    PopulationSize = 2000,      // Размер популяции
    MaxGenerations = 500,       // Макс. поколений
    MaxDepth = 4,               // Макс. глубина дерева
    CrossoverRate = 0.8,        // Вероятность скрещивания
    MutationRate = 0.9,         // Вероятность мутации
    ParsimonyCoefficient = 0.01 // Коэффициент простоты
};
```

## 📈 Пример результата

Библиотека может находить сложные формулы:
- `add(mul(x, x), sub(1, mul(2, x)))` → x² - 2x + 1
- `mul(sin(x), cos(x))` → sin(x)*cos(x)
- `div(1, add(1, exp(neg(x))))` → 1/(1 + e⁻ˣ)

## 🎯 Особенности

- **Защищенные операции** - избегает деления на ноль и других ошибок
- **Множество функций** - +, -, *, /, sin, cos, log, exp, pow и другие
- **Визуализация** - интерактивные графики в HTML отчетах
- **Гибкость** - легко расширяется новыми функциями
