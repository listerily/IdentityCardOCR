package com.example.identity_card_ocr;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteStatement;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import com.example.identity_card_ocr.databinding.ActivityCaptureBinding;
import com.google.common.util.concurrent.ListenableFuture;


import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class CaptureActivity extends AppCompatActivity {
    private ActivityCaptureBinding binding;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private final static int CAMERA_PERMISSION_REQUEST_CODE = 114514;
    private ImageCapture imageCapture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCaptureBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        int frameWidth = binding.frameLayoutCamera.getWidth();
        int frameHeight = (int) Math.round(frameWidth * 0.63084);
        binding.frameLayoutCamera.setMinimumHeight(frameHeight);
        binding.captureButton.setOnClickListener(view -> captureAndRecognize());
        requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            boolean granted = false;
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
                    imageCapture = new ImageCapture.Builder().build();
                    CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(CaptureActivity.this, selector, imageCapture, preview);
                } catch (ExecutionException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    public final MediaType mediaType = MediaType.parse("application/text");
    public static final String postUrl = "http://39.106.132.24:8080/api";

    private void captureAndRecognize() {
        File picturePath = new File(CaptureActivity.this.getFilesDir(), "picture.jpeg");
        ImageCapture.OutputFileOptions outputFileOptions =
                new ImageCapture.OutputFileOptions.Builder(picturePath).build();
        imageCapture.takePicture(outputFileOptions, ContextCompat.getMainExecutor(CaptureActivity.this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        String encodedImage;
                        try {
                            BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(picturePath));
                            byte[] bytes = new byte[inputStream.available()];
                            inputStream.read(bytes, 0, bytes.length);
                            inputStream.close();
                            encodedImage = Base64.encodeToString(bytes, Base64.DEFAULT);
                        } catch (IOException e) {
                            e.printStackTrace();
                            CaptureActivity.this.runOnUiThread(() -> Toast.makeText(CaptureActivity.this,
                                    R.string.capture_failed,
                                    Toast.LENGTH_LONG).show());
                            return;
                        }
                        OkHttpClient client = new OkHttpClient.Builder()
                                .connectTimeout(10, TimeUnit.SECONDS)
                                .writeTimeout(10, TimeUnit.SECONDS)
                                .readTimeout(30, TimeUnit.SECONDS).build();
                        RequestBody body = RequestBody.create(encodedImage, mediaType);
                        Request request = new Request.Builder()
                                .url(postUrl)
                                .post(body)
                                .build();

                        client.newCall(request).enqueue(new Callback() {
                            @Override
                            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                                e.printStackTrace();
                                call.cancel();
                                CaptureActivity.this.runOnUiThread(() -> Toast.makeText(CaptureActivity.this,
                                        R.string.network_failed,
                                        Toast.LENGTH_LONG).show());
                            }

                            @Override
                            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                                try {
                                    if (response.code() != 200) {
                                        throw new IOException("Request failed.");
                                    }
                                    JSONObject jsonObject = new JSONObject(Objects.requireNonNull(response.body()).string());
                                    int success = jsonObject.getInt("success");
                                    if (success == 0) {
                                        throw new IllegalStateException("Failed to analyse ID Card.");
                                    }
                                    String number = jsonObject.getString("number");
                                    String birthYear = jsonObject.getString("year");
                                    String birthMonth = jsonObject.getString("month");
                                    String birthDay = jsonObject.getString("date");
                                    String name = jsonObject.getString("name");
                                    String gender = jsonObject.getString("gender");
                                    String nationality = jsonObject.getString("nationality");
                                    String address = jsonObject.getString("address");
                                    SQLiteDatabase database = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);
                                    database.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
                                    SQLiteStatement statement = database.compileStatement("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?);");
                                    statement.bindString(1, number);
                                    statement.bindString(2, name);
                                    statement.bindString(3, nationality);
                                    statement.bindString(4, gender);
                                    statement.bindString(5, birthYear);
                                    statement.bindString(6, birthMonth);
                                    statement.bindString(7, birthDay);
                                    statement.bindString(8, address);
                                    statement.executeInsert();
                                    database.close();
                                    CaptureActivity.this.runOnUiThread(() -> {
                                        Toast.makeText(CaptureActivity.this,
                                                R.string.recognize_succeed,
                                                Toast.LENGTH_LONG).show();
                                        CaptureActivity.this.finish();
                                    });
                                } catch (JSONException | IOException | IllegalStateException e) {
                                    e.printStackTrace();
                                    CaptureActivity.this.runOnUiThread(() -> {
                                        Toast.makeText(CaptureActivity.this,
                                                R.string.recognize_failed,
                                                Toast.LENGTH_LONG).show();
                                    });
                                }
                            }
                        });
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException error) {
                        error.printStackTrace();
                        CaptureActivity.this.runOnUiThread(() -> Toast.makeText(CaptureActivity.this,
                                R.string.capture_failed,
                                Toast.LENGTH_LONG).show());
                    }
                }
        );
    }
}