package com.example.identity_card_ocr;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import androidx.core.content.FileProvider;
import androidx.navigation.ui.AppBarConfiguration;

import com.example.identity_card_ocr.databinding.ActivityMainBinding;
import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.net.URLConnection;


public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.content.buttonAbout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setMessage(R.string.content_info)
                        .setTitle(R.string.action_show_info)
                        .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {

                            }
                        });
                AlertDialog dialog = builder.create();
                dialog.show();
            }
        });

        binding.content.buttonStartCapturing.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, CaptureActivity.class);
                startActivity(intent);
            }
        });

        binding.content.buttonViewResults.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, ResultsActivity.class);
                startActivity(intent);
            }
        });

        binding.content.buttonExportResults.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                File directory = getExternalCacheDir();
                directory.mkdirs();
                File file = new File(directory, "exported_results.csv");
                try {
                    file.createNewFile();
                    CSVWriter csvWrite = new CSVWriter(new FileWriter(file));
                    SQLiteDatabase db = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);;
                    db.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
                    Cursor curCSV = db.rawQuery("SELECT * FROM results;", null);
                    csvWrite.writeNext(curCSV.getColumnNames());
                    while (curCSV.moveToNext()) {
                        String[] arrStr = {curCSV.getString(0), curCSV.getString(1), curCSV.getString(2),
                                curCSV.getString(3), curCSV.getString(4), curCSV.getString(5),
                                curCSV.getString(6), curCSV.getString(7)};
                        csvWrite.writeNext(arrStr);
                    }
                    csvWrite.close();
                    curCSV.close();
                    db.close();
                } catch (Exception e) {
                    Log.e("MainActivity", "Failed", e);
                    Toast.makeText(MainActivity.this, R.string.export_failed, Toast.LENGTH_LONG).show();
                    return;
                }

                Intent intentShareFile = new Intent(Intent.ACTION_SEND);
                Uri uri = FileProvider.getUriForFile(getApplicationContext(), getPackageName() + ".FileProvider", file);
                intentShareFile.setDataAndType(uri, URLConnection.guessContentTypeFromName(file.getName()));
                intentShareFile.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                intentShareFile.putExtra(Intent.EXTRA_STREAM, uri);
                startActivity(Intent.createChooser(intentShareFile, getString(R.string.action_export_results)));
            }
        });
    }
}