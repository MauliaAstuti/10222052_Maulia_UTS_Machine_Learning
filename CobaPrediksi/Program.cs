using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Collections.Generic;
using System.Linq; // <-- Penting untuk .First()

// 1. Definisikan struktur data input
// PERHATIKAN: Kita tidak lagi pakai 'ImagePath'. 
// Kita akan menyediakan 'Image' (byte array) secara langsung.
public class ModelInput
{
    // Nama 'Image' HARUS SAMA PERSIS dengan 'FeatureColumnName' 
    // yang Anda gunakan saat training.
    [ColumnName("Image")]
    public byte[] Image { get; set; } = Array.Empty<byte>();

    public string Label { get; set; } = string.Empty;
}

// 2. Definisikan struktur data output (prediksi)
public class ModelOutput
{
    // "PredictedLabel" HARUS SAMA PERSIS dengan pipeline LATIHAN Anda
    [ColumnName("PredictedLabel")]
    public string Prediction { get; set; } = string.Empty;
    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Tentukan path ke model dan gambar uji
        string modelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");
        
        // --- GANTI NAMA FILE INI DENGAN NAMA GAMBAR UJI ANDA ---
        string imageToPredictPath = Path.Combine(Environment.CurrentDirectory, "test.jpg"); 
        // ----------------------------------------------------

        Console.WriteLine($"Memuat model dari: {modelPath}");
        
        // 1. Muat (Load) model .zip
        ITransformer model = mlContext.Model.Load(modelPath, out var modelSchema);

        Console.WriteLine($"\nMemprediksi gambar: {Path.GetFileName(imageToPredictPath)}...");

        // 2. BUAT DATA INPUT DENGAN MEMBACA BYTE GAMBAR SECARA MANUAL
        var inputData = new List<ModelInput>()
        {
            new ModelInput
            {
                // Baca file gambar dan ubah menjadi byte array
                Image = File.ReadAllBytes(imageToPredictPath),
                Label = "" // Label dikosongkan
            }
        };

        // 3. Ubah list data menjadi IDataView
        var dataView = mlContext.Data.LoadFromEnumerable(inputData);

        // 4. JALANKAN TRANSFORMASI (PREDIKSI)
        // Sekarang, model akan "melihat" kolom 'Image' (byte[]) yang dibutuhkannya.
        var predictions = model.Transform(dataView);

        // 5. Ambil hasil prediksi dari IDataView
        var predictionResult = mlContext.Data
            .CreateEnumerable<ModelOutput>(predictions, reuseRowObject: false)
            .First(); // Ambil hasil pertama (dan satu-satunya)

        // 6. Tampilkan Hasil
        Console.WriteLine("------------------------------");
        Console.WriteLine($"✅ Hasil Prediksi: {predictionResult.Prediction}");
        Console.WriteLine("------------------------------");
    }
}