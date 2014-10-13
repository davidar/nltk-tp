using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml.Linq;
using log4net;
using Palmmedia.ReportGenerator.Parser.Analysis;
using Palmmedia.ReportGenerator.Properties;

namespace Palmmedia.ReportGenerator.Parser
{
    /// <summary>
    /// Parser for XML reports generated by CodeCoverage.exe.
    /// </summary>
    public class DynamicCodeCoverageParser : ParserBase
    {
        /// <summary>
        /// The Logger.
        /// </summary>
        private static readonly ILog Logger = LogManager.GetLogger(typeof(DynamicCodeCoverageParser));

        /// <summary>
        /// Initializes a new instance of the <see cref="DynamicCodeCoverageParser"/> class.
        /// </summary>
        /// <param name="report">The report file as XContainer.</param>
        public DynamicCodeCoverageParser(XContainer report)
        {
            if (report == null)
            {
                throw new ArgumentNullException("report");
            }

            var modules = report.Descendants("module")
                .OrderBy(m => m.Attribute("name").Value)
                .ToArray();

            Parallel.ForEach(modules, assembly => this.AddAssembly(ProcessAssembly(assembly)));
        }

        /// <summary>
        /// Processes the given assembly.
        /// </summary>
        /// <param name="module">The module.</param>
        /// <returns>The <see cref="Assembly"/>.</returns>
        private static Assembly ProcessAssembly(XElement module)
        {
            string assemblyName = module.Attribute("name").Value;

            Logger.DebugFormat("  " + Resources.CurrentAssembly, assemblyName);

            var classNames = module
                .Elements("functions")
                .Elements("function")
                .Select(f => f.Attribute("type_name").Value)
                .Where(c => !c.Contains("__")
                    && !c.Contains("<")
                    && !c.Contains("."))
                .Distinct()
                .OrderBy(name => name)
                .ToArray();

            var assembly = new Assembly(assemblyName);

            Parallel.ForEach(classNames, className => assembly.AddClass(ProcessClass(module, assembly, className)));

            return assembly;
        }

        /// <summary>
        /// Processes the given class.
        /// </summary>
        /// <param name="module">The module.</param>
        /// <param name="assembly">The assembly.</param>
        /// <param name="className">Name of the class.</param>
        /// <returns>The <see cref="Class"/>.</returns>
        private static Class ProcessClass(XElement module, Assembly assembly, string className)
        {
            var fileIdsOfClass = module
                .Elements("functions")
                .Elements("function")
                .Where(c => c.Attribute("type_name").Value.Equals(className))
                .Elements("ranges")
                .Elements("range")
                .Select(r => r.Attribute("source_id").Value)
                .Distinct();

            var @class = new Class(className, assembly);

            var files = module.Elements("source_files").Elements("source_file");

            foreach (var fileId in fileIdsOfClass)
            {
                string file = files.First(f => f.Attribute("id").Value == fileId).Attribute("path").Value;
                @class.AddFile(ProcessFile(module, fileId, @class, file));
            }

            return @class;
        }

        /// <summary>
        /// Processes the file.
        /// </summary>
        /// <param name="module">The module.</param>
        /// <param name="fileId">The file id.</param>
        /// <param name="class">The class.</param>
        /// <param name="filePath">The file path.</param>
        /// <returns>The <see cref="CodeFile"/>.</returns>
        private static CodeFile ProcessFile(XElement module, string fileId, Class @class, string filePath)
        {
            var methods = module
                .Elements("functions")
                .Elements("function")
                .Where(c => c.Attribute("type_name").Value.StartsWith(@class.Name, StringComparison.Ordinal))
                .Where(m => m.Elements("ranges").Elements("range").Any(r => r.Attribute("source_id").Value == fileId))
                .ToArray();

            SetMethodMetrics(methods, @class);

            var linesOfFile = methods
                .Elements("ranges")
                .Elements("range")
                .Select(l => new
                {
                    LineNumber = int.Parse(l.Attribute("start_line").Value, CultureInfo.InvariantCulture),
                    Coverage = l.Attribute("covered").Value.Equals("no") ? 0 : 1
                })
                .OrderBy(seqpnt => seqpnt.LineNumber)
                .ToArray();

            int[] coverage = new int[] { };

            if (linesOfFile.Length > 0)
            {
                coverage = new int[linesOfFile[linesOfFile.LongLength - 1].LineNumber + 1];

                for (int i = 0; i < coverage.Length; i++)
                {
                    coverage[i] = -1;
                }

                foreach (var seqpnt in linesOfFile)
                {
                    coverage[seqpnt.LineNumber] = coverage[seqpnt.LineNumber] == -1 ? seqpnt.Coverage : Math.Min(coverage[seqpnt.LineNumber] + seqpnt.Coverage, 1);
                }
            }

            return new CodeFile(filePath, coverage);
        }

        /// <summary>
        /// Extracts the metrics from the given <see cref="XElement">XElements</see>.
        /// </summary>
        /// <param name="methods">The methods.</param>
        /// <param name="class">The class.</param>
        private static void SetMethodMetrics(IEnumerable<XElement> methods, Class @class)
        {
            foreach (var method in methods)
            {
                string methodName = method.Attribute("name").Value;

                // Exclude properties and lambda expressions
                if (methodName.StartsWith("get_", StringComparison.Ordinal)
                    || methodName.StartsWith("set_", StringComparison.Ordinal)
                    || Regex.IsMatch(methodName, "<.+>.+__"))
                {
                    continue;
                }

                var metrics = new[] 
                {
                    new Metric(
                        "Blocks covered", 
                        int.Parse(method.Attribute("blocks_covered").Value, CultureInfo.InvariantCulture)),
                    new Metric(
                        "Blocks not covered", 
                        int.Parse(method.Attribute("blocks_not_covered").Value, CultureInfo.InvariantCulture))
                };

                @class.AddMethodMetric(new MethodMetric(methodName, metrics));
            }
        }

    }
}
