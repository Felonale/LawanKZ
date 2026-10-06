using System;
using System.IO;
using System.Runtime.InteropServices;

public static class SofficeWrapper
{
    public static int Main(string[] args)
    {
        string outDir = null;
        string input = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--outdir" && i + 1 < args.Length)
                outDir = args[++i];
            else if (!args[i].StartsWith("-"))
                input = args[i];
        }
        if (String.IsNullOrWhiteSpace(outDir) || String.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Expected --outdir and input document");
            return 2;
        }

        object app = null;
        object doc = null;
        try
        {
            Directory.CreateDirectory(outDir);
            Type wordType = Type.GetTypeFromProgID("Word.Application");
            if (wordType == null)
                throw new InvalidOperationException("Microsoft Word is not installed");
            dynamic word = Activator.CreateInstance(wordType);
            app = word;
            word.Visible = false;
            word.DisplayAlerts = 0;
            dynamic document = word.Documents.Open(Path.GetFullPath(input), ReadOnly: true, AddToRecentFiles: false);
            doc = document;
            try { document.Fields.Update(); } catch { }
            try
            {
                foreach (dynamic toc in document.TablesOfContents)
                    toc.Update();
            }
            catch { }
            string output = Path.Combine(outDir, Path.GetFileNameWithoutExtension(input) + ".pdf");
            document.ExportAsFixedFormat(output, 17);
            document.Close(false);
            word.Quit(false);
            return File.Exists(output) ? 0 : 3;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.ToString());
            try { if (doc != null) ((dynamic)doc).Close(false); } catch { }
            try { if (app != null) ((dynamic)app).Quit(false); } catch { }
            return 1;
        }
        finally
        {
            if (doc != null && Marshal.IsComObject(doc)) Marshal.FinalReleaseComObject(doc);
            if (app != null && Marshal.IsComObject(app)) Marshal.FinalReleaseComObject(app);
        }
    }
}
