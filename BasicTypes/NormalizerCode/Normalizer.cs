﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using BasicTypes.Diagnostics;
using BasicTypes.Dictionary;
using BasicTypes.Exceptions;
using BasicTypes.Extensions;

namespace BasicTypes.NormalizerCode
{
    /// <summary>
    /// Turns human written text into machine parsable text.
    /// </summary>
    /// <remarks>
    /// Assumes the text is toki pona and correctly punctuated.
    /// 
    /// Detect impossible syntax.
    /// 
    /// Turn ", (prep)" into " ~prep"
    /// 
    /// Detect & hyphenate common compound words.
    /// 
    /// </remarks>
    public class Normalizer
    {

        public static string NormalizeText(string text, Dialect dialect)//= null
        {
            //Console.WriteLine("Before: " + text);
            if (string.IsNullOrWhiteSpace(text))
            {
                return "";
            }

            if (IsNullWhiteOrPunctuation(text))
            {
                return "";
            }

            if (dialect == null)
            {
                dialect = Config.CurrentDialect;
            }

            string normalized = text.Trim(new char[] { '\n', '\r', ' ' });

            //Normalize prepositions to ~, so that we don't have tokens with embedded spaces (e.g. foo, kepeken => [foo],[, kepeken])


            bool hasErrors = DetectErrors(normalized, dialect.ThrowOnSyntaxError);
            if (hasErrors)
            {
                normalized = ApplyNormalization(normalized, "Particles", RepairErrors);
            }

            ////"/\\*.*?\\*/"
            if (normalized.Contains("/*") || normalized.Contains("*/"))
            {
                normalized = ApplyNormalization(normalized, "Comments", StripMultilineComments);
            }


            if (dialect.InferCompoundsPrepositionsForeignText)
            {
                normalized = ApplyNormalization(normalized, "Foreign", NormalizeChaos.Normalize);
            }

            //Hyphenated words. This could cause a problem for compound words that cross lines.
            if (normalized.Contains("-\n"))
            {
                normalized = normalized.Replace("-\n", "");
            }

            //can't cope with line breaks.
            if (normalized.Contains("\n"))
            {
                normalized = normalized.Replace("\n", " ");
            }
            if (normalized.Contains("\t"))
            {
                normalized = normalized.Replace("\t", " ");
            }

            //must be after - processing
            if (dialect.InferNumbers)
            {
                normalized = ApplyNormalization(normalized, "Numbers", NormalizeNumbers.FindNumbers);
            }


            if (normalized.Contains("\""))
            {
                normalized = ApplyNormalization(normalized, "ForeignSpace", ProcessWhiteSpaceInForeignText);
            }

            //Extraneous punctuation-- TODO, expand to most other symbols.
            if (normalized.Contains("(") || normalized.Contains(")"))
            {
                normalized = normalized.Replace("(", "");
                normalized = normalized.Replace(")", "");
            }

            //Extraneous commas
            if (normalized.Contains(","))
            {
                normalized = ApplyNormalization(normalized, "ExtraCommas", ProcessExtraneousCommas);
            }

            //Left overs from initial parsing.
            if (normalized.Contains("[NULL]"))
            {
                normalized = normalized.Replace("[NULL]", "");
            }

            if (normalized.Contains(" "))
            {
                normalized = ApplyNormalization(normalized, "ExtraWhiteSpace", ProcessExtraneousWhiteSpace);
            }

            //Okay, phrases should be recognizable now.
            if (dialect.InferCompoundsPrepositionsForeignText)
            {
                normalized = ApplyNormalization(normalized, "Compounds", ProcessCompoundWords);
            }


            string[] preps = Particles.Prepositions;

            bool isPunctuated = false;

            if (!dialect.InferCompoundsPrepositionsForeignText)
            {
                isPunctuated = true;
            }
            foreach (string prep in preps)
            {
                if (normalized.Contains(prep))
                {
                    isPunctuated = normalized.Contains(", " + prep) || normalized.Contains("," + prep);
                    normalized = Regex.Replace(normalized, "," + prep, " ~" + prep);
                    normalized = Regex.Replace(normalized, ", " + prep, " ~" + prep);

                    if (normalized.Contains("~ ~"))
                    {
                        throw new InvalidOperationException(text);
                    }
                }
            }
            if (!isPunctuated && !normalized.Contains("~"))
            {
                foreach (string prep in preps)
                {
                    if (normalized.Contains(prep))
                    {
                        //HACK:Naive repair-- doesn't deal with unusal white space patterns.
                        normalized = Regex.Replace(normalized, " " + prep, " ~" + prep);
                    }
                }
            }



            //la o
            //invisible implicit subject.
            if (normalized.Contains(" la o "))
            {
                normalized = normalized.Replace(" la o ", " la jan Sanwan o ");
            }



            //TODO: detect start of sentence & replace mi X and sina Y with 

            if (normalized.Contains("mi"))
            {
                normalized = ApplyNormalization(normalized, "mi li", ProcessMi);
            }

            if (normalized.Contains("sina"))
            {
                normalized = ApplyNormalization(normalized, "sina li", ProcessSina);
            }


            Dictionary<string, string> pronounModifiersMap = new Dictionary<string, string>
            {
                {"mi wan li ","mi li wan li "},
                {"mi tu li ","mi li tu li "},
                {"mi mute li ","mi li mute li "},
                {"mi suli li ","mi li suli li "},
                
                {"sina wan li ","sina li wan li "},
                {"sina tu li ","sina li tu li "},
                {"sina mute li ","sina li mute li "},
                {"sina suli li ","sina li suli li "},
                
                {"mi en sina li ","mi li en sina li"}
            };

            //undo overnormalization
            foreach (KeyValuePair<string, string> pair in pronounModifiersMap)
            {
                if (normalized.Contains(pair.Value))
                {
                    normalized = normalized.Replace(pair.Value, pair.Key);
                }
            }



            if (normalized.Contains("la mi"))
            {
                bool dontTouch = false;
                foreach (string pronounModifier in pronounModifiers)
                {
                    if (normalized.Contains("la " + pronounModifier))
                    {
                        dontTouch = true;
                    }
                }
                if (!dontTouch)
                {
                    normalized = Regex.Replace(normalized, @"\bla mi\b", "la mi li"); //normalize contractions

                    //If original was, say, "kin la mi li pali", we get a double li li
                    if (normalized.Contains(" li li "))
                    {
                        //undo doubling.
                        normalized = Regex.Replace(normalized, @"\bli li\b", "li"); //normalize contractions
                    }
                }


            }


            if (normalized.Contains("la sina"))
            {
                bool dontTouch = false;
                foreach (string pronounModifier in pronounModifiers)
                {
                    if (normalized.Contains("la " + pronounModifier))
                    {
                        dontTouch = true;
                    }
                }
                if (!dontTouch)
                {
                    normalized = Regex.Replace(normalized, @"\bla sina\b", "la sina li"); //normalize contractions

                    //If original was, say, "kin la sina li pali", we get a double li li
                    if (normalized.Contains(" li li "))
                    {
                        //undo doubling.
                        normalized = Regex.Replace(normalized, @"\bli li\b", "li"); //normalize contractions
                    }
                }

            }

            if (normalized.Contains("~"))
            {
                normalized = ThoseArentPrepositions(preps, normalized);
            }

            normalized = Regex.Replace(normalized, @"^\s+|\s+$", ""); //Remove extraneous whitespace

            //If it is a sentence fragment, I really can't deal with prep phrase that may or may not be in it.
            if (normalized.Contains("~")
                && !normalized.Contains(" li ") //full sentence okay
                && !normalized.StartCheck("o ") //imperative okay
                )
            {
                normalized = normalized.Replace("~", ""); //HACK: This may erase ~ added by user at the start?
            }

            string fakePredicate = " li ijo Nanunanuwakawakawawa.";
            //normalized = UseDummyPredicateForObviousFragments(normalized, fakePredicate);
            //vocatives & exclamations are expected to be fragmentary.


            //It's a prep, but missing the li.
            if (normalized.Contains("sina ~tawa"))
            {
                normalized = normalized.Replace("sina ~tawa", " sina li ~tawa");
            }
            if (normalized.Contains("mi ~tawa"))
            {
                normalized = normalized.Replace("mi ~tawa", "mi li ~tawa");
            }
            //"mi li ~tawa lon" -- the other one is a prep
            if (normalized.Contains("mi li ~tawa lon"))
            {
                normalized = normalized.Replace("mi li ~tawa lon", "mi li tawa ~lon");
            }

            //missing li (ha,previously I skiped this on purpose!)
            if (normalized.Contains("taso, sina soweli"))
            {
                normalized = normalized.Replace("taso, sina soweli", "taso, sina li soweli");
            }
            //overnormalized... mi li ~tawa
            if (normalized.Contains("e mi li ~tawa"))
            {
                normalized = normalized.Replace("e mi li ~tawa", "e mi ~tawa");
            }

            //overnormalized... mama pi mi mute o
            if (normalized.Contains("mama pi mi li mute o"))
            {
                normalized = normalized.Replace("mama pi mi li mute o", "mama pi mi mute o");
            }
            //overnorm
            if (normalized.Contains("sina li ~tawa mi"))
            {
                normalized = normalized.Replace("sina li ~tawa mi", "sina ~tawa mi");
            }
            if (normalized.Contains("sina li o "))
            {
                normalized = normalized.Replace("sina li o ", "sina o ");
            }

            if (text.StartCheck("mi la ") && normalized.StartCheck("mi li "))
            {
                //ugh what a hack.
                normalized = "mi la " + normalized.Substring(9);
            }

            if (normalized.StartCheck("sina li en "))
            {
                //ugh what a hack.
                normalized = "sina en " + normalized.Substring("sina li en ".Length);
            }

            if (normalized.StartCheck("mi li en "))
            {
                //ugh what a hack.
                normalized = "mi en " + normalized.Substring("mi li en ".Length);
            }

            //e mi li mute
            if (normalized.Contains("e mi li mute"))
            {
                normalized = normalized.Replace("e mi li mute", "e mi mute");
            }


            //One off that comes back?
            foreach (string oneOff in new string[] {
                                                         "li ~lon poka e", //place something next to
                                                         "li ~tawa tu e"
                                                    })
            {
                normalized = normalized.Replace(oneOff, oneOff.Replace("~", ""));
            }

            //Probably added above by mistake
            while (normalized.Contains("  "))
            {
                normalized = normalized.Replace("  ", " ");
            }

            if (normalized.Contains("'"))
            {
                normalized = ApplyNormalization(normalized, "DirectQuotes", AddDirectedQuotes);
            }

            if (normalized == fakePredicate)
            {
                throw new InvalidOperationException("started with " + text);
            }
            if (normalized.StartCheck("« »"))
            {
                throw new InvalidOperationException("quote recognition went wrong: " + text);
            }
            return normalized;
        }

        private static string StripMultilineComments(string arg)
        {
            //https://stackoverflow.com/questions/13014947/regex-to-match-a-c-style-multiline-comment
            return Regex.Replace(arg, "/\\*.*?\\*/", "");
        }

        private static string[] pronounModifiers = new string[]
            {
                "mi wan",
                "mi tu",
                "mi mute",
                "mi suli",

                "sina wan",
                "sina tu",
                "sina mute",
                "sina suli",

                "mi en sina",
            };

        private static string ProcessSina(string normalized)
        {
            normalized = NormalizedSinaLi(normalized);
            normalized = NormalizedSinaLi(normalized, "'");
            normalized = NormalizedSinaLi(normalized, "«");
            normalized = NormalizedSinaLi(normalized, "' ");
            normalized = NormalizedSinaLi(normalized, "« ");
            normalized = NormalizedSinaLi(normalized, "taso "); //conjunctions behave like leading punct


            foreach (Word possible in Words.Dictionary.Values)
            {
                //Need a new concept-- ordinary POS & fallback POS
                string pronoun = "sina";
                Dictionary<string, Dictionary<string, string[]>> glosses;
                if (Words.Glosses.TryGetValue(possible.Text, out glosses))
                {
                    Dictionary<string, string[]> enGlosses;
                    if (glosses.TryGetValue("en", out enGlosses))
                    {
                        if ((enGlosses.ContainsKey("vt") && enGlosses["vt"].Length > 0)
                            ||
                            (enGlosses.ContainsKey("vi") && enGlosses["vi"].Length > 0))
                        {
                            string unLied = " " + pronoun + " " + possible + " ";
                            if (normalized.Contains(unLied) && !pronounModifiers.Contains(unLied.Trim()))
                            {
                                normalized = normalized.Replace(unLied, " " + pronoun + " li " + possible + " ");
                            }
                        }
                    }
                }
            }
            return normalized;
        }

        private static string ProcessMi(string normalized)
        {
            if (normalized.Contains("mi wile ala e ma li"))
            {
                Console.WriteLine("Ok");
            }
            normalized = NormalizedMiLi(normalized);
            normalized = NormalizedMiLi(normalized, "'");
            normalized = NormalizedMiLi(normalized, "«");
            normalized = NormalizedMiLi(normalized, "' ");
            normalized = NormalizedMiLi(normalized, "« ");

            foreach (Word possible in Words.Dictionary.Values)
            {
                //Need a new concept-- ordinary POS & fallback POS
                string pronoun = "mi";
                Dictionary<string, Dictionary<string, string[]>> glosses;
                if (Words.Glosses.TryGetValue(possible.Text, out glosses))
                {
                    Dictionary<string, string[]> enGlosses;
                    if (glosses.TryGetValue("en", out enGlosses))
                    {
                        if ((enGlosses.ContainsKey("vt") && enGlosses["vt"].Length > 0)
                            ||
                            (enGlosses.ContainsKey("vi") && enGlosses["vi"].Length > 0))
                        {
                            string unLied = " " + pronoun + " " + possible + " ";
                            if (normalized.Contains(unLied) && !pronounModifiers.Contains(unLied.Trim()))
                            {
                                normalized = normalized.Replace(unLied, " " + pronoun + " li " + possible + " ");
                            }
                        }
                    }
                }
            }
            return normalized;
        }

        private static string ProcessExtraneousCommas(string normalized)
        {
            foreach (string s in new String[] { ", li ", ", la ", ",la ", " la, ", " la,", ", o ", ",o " })
            {
                if (normalized.Contains(s))
                {
                    normalized = normalized.Replace(s, " " + s.Trim(new char[] { ' ', ',' }) + " ");
                }
            }
            return normalized;
        }

        private static string ApplyNormalization(string normalized, string what, Func<string, string> normalization)
        {
            string copy = string.Copy(normalized);
            normalized = normalization.Invoke(normalized);// NormalizeChaos.Normalize(normalized);
            if (copy != normalized)
            {
                Tracers.Normalize.TraceInformation(what + "1:" + copy);
                Tracers.Normalize.TraceInformation(what + "2:" + normalized);
            }
            return normalized;
        }

        private static string ProcessCompoundWords(string normalized)
        {
            //Because profiler said these were called a lot 
            string[] modalsDash = Token.Modals.Select(x => x + "-").ToArray();
            string[] prepsDash = Particles.Prepositions.Select(x => x + "-").ToArray();
            string[] dashPreps = Particles.Prepositions.Select(x => x + "-").ToArray();

            foreach (var pair in CompoundWords.Dictionary.OrderBy(x => x.Key.Length * -1))
            {
                //Need to treat la fragments separately.
                if (pair.Key.EndCheck("-la")) continue;
                if (pair.Key.StartCheck("pi-")) continue; //This is essentially a possessive or adjective noun phrase, should deal with via POS for compound words.
                if (pair.Key.EndCheck("-ala")) continue; //negation is special.
                if (pair.Key.StartCheck("li-")) continue; //These are essentially verb markers and all verbs phrases are templates (i.e. can have additional words inserted & be the same template)
                if (pair.Key.Contains("-e-")) continue;

                
                //Weak compounds, i.e. to close to noun + ordinary modifier because the modifier is excedingly common.
                //if (pair.Key.EndCheck("-lili")) continue; //jan lili = child?  Ugh, but it is so common.
                //if (pair.Key.EndCheck("-pona")) continue;//jan pona...

                //TODO: Can't distinguish these yet.
                //ijo li wile sona e ni. //modal + verb
                //mi pana e wile-sona ni.//question
                bool thatsAModal = false;
                foreach (string modal in modalsDash)
                {
                    if (pair.Key.StartCheck(modal))
                    {
                        thatsAModal = true;
                    }
                }

                if (thatsAModal) continue;

                //lon-poka 
                bool thatsAPreposition = false;

                //Perf trace said concats were expensive.
                foreach (string preposition in dashPreps)
                {
                    if (pair.Key.EndCheck(preposition)) // jan ante li toki kepeken toki ante taso.
                    {
                        thatsAPreposition = true;
                    }
                }
                foreach (string preposition in prepsDash)
                {
                    if (pair.Key.StartCheck(preposition))
                    {
                        thatsAPreposition = true;
                    }

                }
                if (thatsAPreposition) continue;


                string spacey = pair.Key.Replace("-", " ");
                if (normalized.Contains(spacey))
                {
                    string savePoint = string.Copy(normalized);
                    Regex r = new Regex(@"\b" + spacey + @"\b");
                    var isIt = r.Match(normalized);
                    if (isIt.Success)
                    {
                        //This won't replace on boundaries though.
                        normalized = normalized.Replace(spacey, pair.Key.Trim(new char[] { ' ', '-' }));
                    }
                    if (normalized.Contains("-" + pair.Key) || normalized.Contains(pair.Key + "-"))
                    {
                        //Undo! We matched a word that crosses compound words. How is that even possible?
                        normalized = savePoint;
                    }
                }
            }
            return normalized;
        }

        private static string ProcessExtraneousWhiteSpace(string normalized)
        {
            //Line breaks & other white spaces make it harder to find boundaries
            //jan li
            //tawa. ==> jan li tawa.
            normalized = normalized.Replace("\n", " ");


            //Extraneous whitespace
            while (normalized.Contains("  "))
            {
                normalized = normalized.Replace("  ", " ");
            }
            return normalized;
        }

        private static string AddDirectedQuotes(string normalized)
        {
            StringBuilder sb = new StringBuilder();
            bool open = true;
            foreach (char c in normalized)
            {
                if (c == '\'')
                {
                    if (open)
                    {
                        sb.Append("«");
                    }
                    else
                    {
                        sb.Append("»");
                    }
                    open = !open;
                }
                else
                {
                    sb.Append(c);
                }
            }
            normalized = sb.ToString();
            return normalized;
        }

        private static string UseDummyPredicateForObviousFragments(string normalized, string fakePredicate)
        {
            if (!(normalized.Contains(" li ")) && !IsVocative(normalized) && !IsExclamatory(normalized) &&
                !IsFragment(normalized))
            {
                //Add a marker that we can later remove. Allows for parsing NPs, like titles, as if
                //they were sentences.
                if (normalized.EndCheck("."))
                {
                    normalized = normalized.Substring(0, normalized.Length - 1) + fakePredicate;
                }
                else
                {
                    normalized = normalized + fakePredicate;
                }
            }
            return normalized;
        }

        public static bool IsNullWhiteOrPunctuation(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return true;


            foreach (char c in value)
            {
                if (!"!.', !@#$%^&*())_[]|~`<>?:\n '»".Contains(c))
                {
                    return false;
                }
            }
            return true;
        }


        private static string ProcessWhiteSpaceInForeignText(string normalized)
        {
            StringBuilder sb = new StringBuilder(normalized.Length);
            bool insideForeignText = false;

            foreach (char c in normalized)
            {
                if (c == '"')
                {
                    insideForeignText = !insideForeignText;
                }
                if (insideForeignText && c == ' ')
                {
                    sb.Append('*');
                }
                else
                {
                    sb.Append(c);
                }
            }
            if (insideForeignText)
            {
                sb.Append('"');
            }
            return sb.ToString();
        }

        private static string ThoseArentPrepositions(string[] preps, string normalized)
        {
            //I don't even know what that means.
            //This is just so corpus texts can pass. Better solution is to write annoted text in the first place.
            foreach (string oneOff in new string[] {
                                                        " li ~tawa wawa e", // motion thingy.  Maybe catch with common adverbs?
                                                        " li ~lon wawa e ",// placement thingy. Quickly place something.
                                                        "tenpo ~tawa ali la", //This is just strange.
                                                        "kon ~tawa seli li",// tawa is normal adjective. 
                                                         "kon ~tawa lete li",// tawa is normal adjective. 
                                                         "kon ~tawa telo li",// tawa is normal adjective. 
                                                         "tenpo ~tawa mi", //Time of my leaving
                                                         "li ~tawa anpa ", //Go down.
                                                         "li ~tawa kon ", //fly
                                                         "li ~tawa sewi ", //go up
                                                         "nasin ~sama ala la",//as for different ways... 
                                                         "li ~sama lili e ",//to copy somewhat
                                                         "kon en ~lon en moli", //existence vs in
                                                         "li ~tawa suli e", //move something
                                                         "mi ~lon poka e", //place something next to
                                                         "li ~lon poka e", //place something next to
                                                         "li ~kepeken sin e", //use again
                                                         "li ~tawa jo wawa e", //I don't even know what this means
                                                         "tenpo pi mi ~lon sike", //my life (span)
                                                         "o ~tawa kon e" // moving stuff around.
                                                    })
            {
                normalized = normalized.Replace(oneOff, oneOff.Replace("~", ""));
            }

            //As in jan sama o!
            //jan ~sama o!
            if (normalized.Contains(" ~sama o "))
            {
                normalized = normalized.Replace(" ~sama o ", " sama o ");
            }
            if (normalized.Contains(" ~sama o, "))
            {
                normalized = normalized.Replace(" ~sama o, ", " sama o, ");
            }
            if (normalized.EndCheck(" ~sama o"))
            {
                normalized = normalized.Replace(" ~sama o", " sama o");
            }
            if (normalized.EndCheck(" ~sama o!"))
            {
                normalized = normalized.Replace(" ~sama o!", " sama o!");
            }
            if (normalized.EndCheck(" ~sama o?"))
            {
                normalized = normalized.Replace(" ~sama o?", " sama o?");
            }
            if (normalized.EndCheck(" ~sama o."))
            {
                normalized = normalized.Replace(" ~sama o.", " sama o.");
            }
            //li kama tawa ==> verb!


            //li ~kepeken e
            foreach (string prep in preps)
            {
                string barePrep = " li ~" + prep + " e ";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }


            //la ~lon 
            foreach (string prep in preps)
            {
                string leadPrep = " la ~" + prep + " ";
                if (normalized.Contains(leadPrep))
                {
                    normalized = normalized.Replace(leadPrep, leadPrep.Replace("~", ""));
                }
            }


            //li ~tawa ala e
            foreach (string prep in preps)
            {
                string barePrep = " li ~" + prep + " ala e ";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }
            //o ~kepeken ala e
            foreach (string prep in preps)
            {
                string barePrep = " o ~" + prep + " ala e ";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }
            //~tawa e
            foreach (string prep in preps)
            {
                string barePrep = " ~" + prep + " e ";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }
            //li ken ~tawa sike e 
            foreach (string prep in preps)
            {
                //TODO: Add other modals.
                //modal + prep => probably verb.
                string barePrep = " li ken ~" + prep;
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }

            //~tan-  it's a prep but inside a compound phrase, we don't care. It's opaque.
            foreach (string prep in preps)
            {
                string barePrep = "~" + prep + "-";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }

            //~tawa pi --- not possible.
            foreach (string prep in preps)
            {
                string barePrep = "~" + prep + " pi ";
                if (normalized.Contains(barePrep))
                {
                    normalized = normalized.Replace(barePrep, barePrep.Replace("~", ""));
                }
            }

            //lon ala li pana e lon
            foreach (string prep in preps)
            {
                string initial = "~" + prep + " ";
                if (normalized.StartCheck(initial))
                {
                    normalized = prep + " " + normalized.Substring((prep + " ").Length);
                    //normalized = normalized.Replace(initial, initial.Replace("~", ""));
                }
            }

            //This is so uncommon, that it actually better to assume it isn't a prep.
            foreach (string prep in preps)
            {
                string piHeadedPrep = " pi ~" + prep + " ";
                if (normalized.Contains(piHeadedPrep))
                {
                    normalized = normalized.Replace(piHeadedPrep, piHeadedPrep.Replace("~", ""));
                }
            }


            //.. li sama.
            foreach (string prep in preps)
            {
                foreach (char c in ":!?.")
                {
                    string terminalPrep = "~" + prep + c;
                    if (normalized.Contains(terminalPrep))
                    {
                        normalized = normalized.Replace(terminalPrep, prep + c);
                    }
                }
            }

            //variation on terminal prep
            //lon lawa sama o tawa...
            foreach (string prep in preps)
            {
                foreach (var predicateSplitter in new string[] { " o ", " li " })
                {
                    string terminalPrep = "~" + prep + predicateSplitter;
                    if (normalized.Contains(terminalPrep))
                    {
                        normalized = normalized.Replace(terminalPrep, prep + predicateSplitter);
                    }
                }
            }


            //la terminates
            foreach (string prep in preps)
            {
                foreach (var predicateSplitter in new string[] { " la " })
                {
                    string terminalPrep = "~" + prep + predicateSplitter;
                    if (normalized.Contains(terminalPrep))
                    {
                        normalized = normalized.Replace(terminalPrep, prep + predicateSplitter);
                    }
                }
            }

            //And again with commas
            foreach (string prep in preps)
            {
                foreach (var predicateSplitter in new string[] { ", o ", ", li " })
                {
                    string terminalPrep = "~" + prep + predicateSplitter;
                    if (normalized.Contains(terminalPrep))
                    {
                        normalized = normalized.Replace(terminalPrep, prep + predicateSplitter);
                    }
                }
            }

            foreach (string prep in preps)
            {
                foreach (var conjunction in new string[] { "anu ", "taso " })
                {
                    string initialPrep = conjunction + "~" + prep;
                    if (normalized.StartCheck(initialPrep))
                    {
                        normalized = normalized.Replace(initialPrep, initialPrep.Replace("~",""));
                    }
                }
            }

            foreach (string prep in preps)
            {
                string terminalPrep = "~" + prep;
                if (normalized.EndCheck(terminalPrep))
                {
                    normalized = normalized.Remove(normalized.LastIndexOf(terminalPrep),1);
                }
            }

            //e ~sama
            foreach (string prep in preps)
            {
                //e
                foreach (char c in "e")
                {
                    string terminalPrep = " " + c + " ~" + prep;
                    if (normalized.Contains(terminalPrep))
                    {
                        normalized = normalized.Replace(terminalPrep, " e " + prep);
                    }
                }
            }

            //~tawa pi jan Puta li pona

            //lon poka ma ni.
            foreach (string prep1 in preps)
            {
                foreach (string prep2 in preps)
                {
                    string doublePrep = "~" + prep1 + " " + "~" + prep2;
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, "~" + prep1 + " " + prep2);
                    }
                    doublePrep = "~" + prep2 + " " + "~" + prep1;
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, "~" + prep2 + " " + prep1);
                    }
                }
            }


            //This could go many directions.  
            //li tawa en tan (lon ...) -- prep phrase like  *** Prep phrase parser blows chunks on these.
            //li tawa en tan li .... -- verb like
            //li tawa en tan e ... -- verb like *edgy but sort of okay
            //li tawa en tan. ... -- predicate like. ** OK as predicate
            foreach (string prep1 in preps)
            {
                foreach (string prep2 in preps)
                {
                    string doublePrep = "~" + prep1 + " en " + "~" + prep2;
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, prep1 + " " + prep2);
                    }
                    doublePrep = "~" + prep2 + " en " + "~" + prep1;
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, prep2 + " " + prep1);
                    }
                }
            }


            //Can't be prep phrase
            //li ~tawa en tan li 

            foreach (string prep1 in preps)
            {
                foreach (string prep2 in preps)
                {
                    string doublePrep = " li ~" + prep1 + " en " + prep2 + " li ";
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, doublePrep.Replace("~", ""));
                    }
                    doublePrep = " li ~" + prep2 + " en " + prep1 + " li ";
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, doublePrep.Replace("~", ""));
                    }
                }
            }


            //li tawa en ~tan li 
            foreach (string prep1 in preps)
            {
                foreach (string prep2 in preps)
                {
                    string doublePrep = " li " + prep1 + " en ~" + prep2 + " li ";
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, doublePrep.Replace("~", ""));
                    }
                    doublePrep = " li " + prep2 + " en ~" + prep1 + " li ";
                    if (normalized.Contains(doublePrep))
                    {
                        normalized = normalized.Replace(doublePrep, doublePrep.Replace("~", ""));
                    }
                }
            }

            //li ~kepeken ala kepeken e
            foreach (string prep in preps)
            {
                string question = " li ~" + prep + " ala " + prep + " e ";
                if (normalized.Contains(question))
                {
                    normalized = normalized.Replace(question, question.Replace("~", ""));
                }
            }


            return normalized;
        }

        private static string NormalizedSinaLi(string normalized, string punctuation = "")
        {
            if (normalized.StartCheck(punctuation + "sina ") && !normalized.StartCheck(punctuation + "sina li "))
            {
                bool possibleProunoun = !normalized.StartCheck(punctuation + "sina suli")
                                        && !normalized.StartCheck(punctuation + "sina mute")
                                        && !normalized.StartCheck(punctuation + "sina tu");
                //mi mute li suli.
                //mi toki.

                if (possibleProunoun && normalized.Contains("sina li " + normalized.Substring(5 + punctuation.Length)))
                {
                    //Skip we probalby have a li already.
                }
                else
                {
                    normalized = punctuation + "sina li " + normalized.Substring(5 + punctuation.Length);
                }
            }
            return normalized;
        }

        private static string NormalizedMiLi(string normalized, string punctuation = "")
        {
            if (normalized.StartCheck(punctuation + "mi ") && !normalized.StartCheck(punctuation + "mi li "))
            {
                //modified mi is rare.
                bool possibleProunoun = !normalized.StartCheck(punctuation + "mi suli")
                                        && !normalized.StartCheck(punctuation + "mi mute")
                                        && !normalized.StartCheck(punctuation + "mi tu");
                //mi mute li suli.
                //mi toki.

                //modified mi will force a li.
                if (possibleProunoun && !normalized.Contains("mi li " + normalized.Substring(3 + punctuation.Length)))
                {
                    //Skip we probalby have a li already.
                }
                //else
                {
                    normalized = punctuation + "mi li " + normalized.Substring(3 + punctuation.Length);
                }
            }
            return normalized;
        }

        private static bool IsFragment(string value)
        {
            string normalized = value.Trim(new char[] { ' ', '\'', '«', '»', '?', '!', '.' });
            return normalized.EndCheck(" la");
        }

        //trying to catch ike a!
        public static bool IsExclamatory(string value)
        {
            string normalized = value.Trim(new char[] { ' ', '\'', '«' });
            return normalized.EndCheck("!");
        }

        //Trying to catch o! jan o! jan o. jan o, o.
        public static bool IsVocative(string value)
        {
            string normalized = value.Trim(new char[] { ' ', '\'', '«', '.', '!', '?', ',' });
            return normalized.Contains(" o ") ||
                   normalized.StartCheck("o ") ||
                   normalized.EndCheck(" o") ||
                   normalized == "o";
        }

        public static string RepairErrors(string phrase)
        {
            foreach (string s1 in new String[] { "li", "la", "e", "pi" })
            {
                phrase = Regex.Replace(phrase, @"\b" + s1 + " " + s1 + @"\b", "[SYNTAX ERROR: " + s1 + " " + s1 + "]");
            }
            foreach (string s1 in new String[] { "li", "la", "e", "pi" })
            {
                foreach (string s2 in new String[] { "li", "la", "e", "pi" })
                {
                    Match found = Regex.Match(phrase, @"\b" + s1 + " " + s2 + @"\b");
                    if (found.Success)
                    {
                        phrase = Regex.Replace(phrase, @"\b" + s1 + " " + s1 + @"\b", "[SYNTAX ERROR: " + s1 + " " + s2 + "]");
                    }
                }
            }

            return phrase;
        }


        public static bool DetectErrors(string phrase, bool throwOnSyntaxErrors = false)
        {

            foreach (string s1 in new String[] { "li", "la", "e", "pi" })
            {
                Match found = Regex.Match(phrase, @"\b" + s1 + " " + s1 + @"\b");
                if (found.Success)
                {
                    if (throwOnSyntaxErrors)
                        throw new DoubleParticleException();
                    else
                    {
                        return true;
                    }
                }
            }
            foreach (string s1 in new String[] { "li", "la", "e", "pi" })
            {
                foreach (string s2 in new String[] { "li", "la", "e", "pi" })
                {
                    Match found = Regex.Match(phrase, @"\b" + s1 + " " + s2 + @"\b");
                    if (found.Success)
                    {
                        if (throwOnSyntaxErrors)
                            throw new TpSyntaxException("Illegal series of particles : " + s1 + " " + s2 + " in " + phrase);
                        else
                        {
                            return true;
                        }
                    }
                }
            }

            return false;

        }
    }
}