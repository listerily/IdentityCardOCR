package com.example.identity_card_ocr;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.widget.Toast;

import com.example.identity_card_ocr.databinding.ActivityCaptureBinding;
import com.google.common.util.concurrent.ListenableFuture;


import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;

public class CaptureActivity extends AppCompatActivity {
    private ActivityCaptureBinding binding;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private final static int CAMERA_PERMISSION_REQUEST_CODE = 114514;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCaptureBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        int frameWidth = binding.frameLayoutCamera.getWidth();
        int frameHeight = (int)Math.round(frameWidth * 0.63084);
        binding.frameLayoutCamera.setMinimumHeight(frameHeight);

        SQLiteDatabase database = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);
        database.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
        database.execSQL("INSERT INTO results VALUES('114514192608172220', '唐可可', '汉', '女', '2002', '08', '17', '天津市津南区下北泽南开大学下北泽校区');");
        database.close();

        requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            boolean granted =  false;
            for (int i = 0; i < permissions.length; i++) {
                String permission = permissions[i];
                int grantResult = grantResults[i];

                if (permission.equals(Manifest.permission.CAMERA)) {
                    granted = grantResult == PackageManager.PERMISSION_GRANTED;
                }
            }
            if (granted) {
                startCamera();
            } else {
                Toast.makeText(this,
                        R.string.camera_permission_denied,
                        Toast.LENGTH_LONG).show();
                finish();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    Preview preview = new Preview.Builder()
                            .build();
                    preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());
                    ImageAnalysis analysis = new ImageAnalysis.Builder()
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build();
                    analysis.setAnalyzer(ContextCompat.getMainExecutor(CaptureActivity.this),
                            new IdentityCardAnalyzer());
                    CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(CaptureActivity.this, selector, analysis, preview);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }
        }, ContextCompat.getMainExecutor(this));

    }

    private class IdentityCardAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy image) {

        }
    }
}