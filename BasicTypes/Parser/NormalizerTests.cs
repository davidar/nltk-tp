﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace BasicTypes.Parser
{
    [TestFixture]
    public class NormalizerTests
    {

        //nena meli kin li tawa en tan, li kama nena pi suli en kiwen.

        [Test]
        public void LiTawaEnTan()
        {
            //sina toki e ni: 
            const string s = "nena meli kin li tawa en tan, li kama nena pi suli en kiwen.";
            Console.WriteLine("Original  : " + s);
            string normalized = Normalizer.NormalizeText(s, Dialect.DialectFactory);
            Console.WriteLine("Normalized: " + normalized);
            //sina li toki e ni: 
            const string expected = "nena meli kin li tawa en tan li kama nena pi suli en kiwen.";
            Assert.AreEqual(expected, normalized);
        }


        [Test]
        public void Normalize_IntransitiveVerb()
        {
            string value = Normalizer.NormalizeText("jan li moku, kepeken ilo moku");
            Console.WriteLine(value);
            Assert.IsTrue(value.Contains("~"), value);
        }

        [Test]
        public void MiWileENi()
        {
            //sina toki e ni: 
            const string s = "mi wile e ni.";


            Console.WriteLine("Original  : " + s);
            string normalized = Normalizer.NormalizeText(s, Dialect.DialectFactory);
            Console.WriteLine("Normalized: " + normalized);
            //sina li toki e ni: 
            const string expected = "mi li wile e ni.";
            Assert.AreEqual(expected, normalized);
        }

        [Test]
        public void LonPokaSentence()
        {
            const string s = "jan Puta li lon poka ma Nepali en Inteja";

            Console.WriteLine("Original  : " + s);
            string normalized = Normalizer.NormalizeText(s, Dialect.DialectFactory);
            Console.WriteLine("Normalized: " + normalized);

            const string expected = "jan Puta li ~lon poka ma Nepali en Inteja";
            Assert.AreEqual(expected,normalized);
        }


        [Test]
        public void SomethingLiSama()
        {
            const string s = "ni li sama.";

            Console.WriteLine("Original  : " + s);
            string normalized = Normalizer.NormalizeText(s, Dialect.DialectFactory);
            Console.WriteLine("Normalized: " + normalized);

            const string expected = "ni li sama.";
            Assert.AreEqual(expected, normalized);
        }

        [Test]
        public void TawaPiJanPutaLiPona()
        {
            const string s = "tawa pi jan Puta li pona.";

            Console.WriteLine("Original  : " + s);
            string normalized = Normalizer.NormalizeText(s, Dialect.DialectFactory);
            Console.WriteLine("Normalized: " + normalized);

            const string expected = "tawa pi jan Puta li pona.";
            Assert.AreEqual(expected, normalized);
        }

    }
}