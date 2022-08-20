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
import android.database.sqlite.SQLiteStatement;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.Toast;

import com.example.identity_card_ocr.databinding.ActivityCaptureBinding;
import com.google.common.util.concurrent.ListenableFuture;


import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.ExecutionException;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCaptureBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        int frameWidth = binding.frameLayoutCamera.getWidth();
        int frameHeight = (int)Math.round(frameWidth * 0.63084);
        binding.frameLayoutCamera.setMinimumHeight(frameHeight);

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
                            .build();
                    analysis.setAnalyzer(ContextCompat.getMainExecutor(CaptureActivity.this),
                            new IdentityCardAnalyzer());
                    CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(CaptureActivity.this, selector, analysis, preview);
                } catch (ExecutionException | InterruptedException e) {
                    e.printStackTrace();
                }

            }
        }, ContextCompat.getMainExecutor(this));

    }

    private class IdentityCardAnalyzer implements ImageAnalysis.Analyzer {
        public final MediaType mediaType = MediaType.parse("application/text");
        public static final String postUrl= "https://reqres.in/api/users/";
        @Override
        public void analyze(@NonNull ImageProxy image) {
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer vuBuffer = image.getPlanes()[2].getBuffer();
            int ySize = yBuffer.remaining();
            int vuSize = vuBuffer.remaining();
            byte[] nv21 = new byte[ySize + vuSize];
            yBuffer.get(nv21, 0, ySize);
            vuBuffer.get(nv21, ySize, vuSize);
            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 50, out);
            byte[] imageBytes = out.toByteArray();
            Bitmap bm = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            bm.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
            byte[] b = outputStream.toByteArray();
            String encodedImage = Base64.encodeToString(b, Base64.DEFAULT);

            OkHttpClient client = new OkHttpClient();
            RequestBody body = RequestBody.create(encodedImage, mediaType);
            Request request = new Request.Builder()
                    .url(postUrl)
                    .post(body)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                    call.cancel();
                    CaptureActivity.this.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(CaptureActivity.this,
                                    R.string.network_failed,
                                    Toast.LENGTH_LONG).show();
                        }
                    });
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException ex) {
                        ex.printStackTrace();
                    }
                    image.close();
                }

                @Override
                public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                    try {
                        JSONObject jsonObject = new JSONObject(Objects.requireNonNull(response.body()).string());
                        int success = jsonObject.getInt("success");
                        if (success == 0) {
                            throw new IOException("Failed to analyse ID Card.");
                        }
                        String number = jsonObject.getString("number");
                        String birthYear = jsonObject.getString("year");
                        String birthMonth = jsonObject.getString("month");
                        String birthDay = jsonObject.getString("ydate");
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
                        CaptureActivity.this.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(CaptureActivity.this,
                                    android.R.string.ok,
                                    Toast.LENGTH_LONG).show();
                        }
                    });
                        CaptureActivity.this.finish();
                    } catch (JSONException e) {
                        e.printStackTrace();
                        Toast.makeText(CaptureActivity.this,
                                R.string.network_failed,
                                Toast.LENGTH_LONG).show();
                    } catch (IOException e) {
                        e.printStackTrace();
                    } finally {
                        image.close();
                    }
                }
            });
        }
    }
}