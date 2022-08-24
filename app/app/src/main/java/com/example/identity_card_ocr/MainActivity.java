package com.example.identity_card_ocr;

import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import androidx.core.content.FileProvider;

import com.example.identity_card_ocr.databinding.ActivityMainBinding;
import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.net.URLConnection;


public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Inflate view using view binding
        com.example.identity_card_ocr.databinding.ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Set onClickListener for buttonAbout
        binding.content.buttonAbout.setOnClickListener(view -> {
            // Show an alert dialog, telling user app info
            AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
            builder.setMessage(R.string.content_info)
                    .setTitle(R.string.action_show_info)
                    .setPositiveButton(android.R.string.ok, (dialogInterface, i) -> {
                        // Do nothing, skipped.
                    });
            AlertDialog dialog = builder.create();
            dialog.show();
        });

        // Set onClickListener for buttonStartCapturing
        binding.content.buttonStartCapturing.setOnClickListener(view -> {
            // Start CaptureActivity
            Intent intent = new Intent(MainActivity.this, CaptureActivity.class);
            startActivity(intent);
        });

        // Set onClickListener for buttonViewResults
        binding.content.buttonViewResults.setOnClickListener(view -> {
            // Start ResultsActivity
            Intent intent = new Intent(MainActivity.this, ResultsActivity.class);
            startActivity(intent);
        });

        // Set onClickListener for buttonExportResults
        binding.content.buttonExportResults.setOnClickListener(view -> {
            // Write captured results to external cache
            File directory = getExternalCacheDir();
            directory.mkdirs();
            File file = new File(directory, "exported_results.csv");
            try {
                file.createNewFile();
                // Create csvWriter for our data export
                CSVWriter csvWrite = new CSVWriter(new FileWriter(file));
                // Open or create SQLiteDatabase
                SQLiteDatabase db = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);
                // Create table if not exists
                db.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
                // Fetching data using select statement
                Cursor curCSV = db.rawQuery("SELECT id_number, name, nationality, gender, birth_year, birth_month, birth_day FROM results;", null);
                csvWrite.writeNext(curCSV.getColumnNames());
                // Fetching data row by row
                while (curCSV.moveToNext()) {
                    String[] arrStr = {curCSV.getString(0), curCSV.getString(1), curCSV.getString(2),
                            curCSV.getString(3), curCSV.getString(4), curCSV.getString(5),
                            curCSV.getString(6)};
                    csvWrite.writeNext(arrStr);
                }
                // Close all
                csvWrite.close();
                curCSV.close();
                db.close();
            } catch (Exception e) {
                Log.e("MainActivity", "Failed", e);
                Toast.makeText(MainActivity.this, R.string.export_failed, Toast.LENGTH_LONG).show();
                return;
            }

            // Create intent and share this file using android FileProvider.
            Intent intentShareFile = new Intent(Intent.ACTION_SEND);
            Uri uri = FileProvider.getUriForFile(getApplicationContext(), getPackageName() + ".FileProvider", file);
            intentShareFile.setDataAndType(uri, URLConnection.guessContentTypeFromName(file.getName()));
            intentShareFile.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intentShareFile.putExtra(Intent.EXTRA_STREAM, uri);
            startActivity(Intent.createChooser(intentShareFile, getString(R.string.action_export_results)));
        });
    }
}