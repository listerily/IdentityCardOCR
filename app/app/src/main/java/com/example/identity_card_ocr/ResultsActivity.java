package com.example.identity_card_ocr;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NavUtils;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.DialogInterface;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.example.identity_card_ocr.databinding.ActivityResultsBinding;

import java.util.ArrayList;
import java.util.Objects;

public class ResultsActivity extends AppCompatActivity {
    private ActivityResultsBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityResultsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setDisplayShowHomeEnabled(true);
            getSupportActionBar().setTitle(R.string.page_title_results);
        }
        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                new AlertDialog.Builder(ResultsActivity.this)
                        .setTitle(R.string.remove_all)
                        .setMessage(R.string.message_remove_all)
                        .setNegativeButton(android.R.string.cancel, new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {

                            }
                        })
                        .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                SQLiteDatabase database = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);
                                database.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
                                database.execSQL("DELETE FROM results;");
                                database.close();
                                initializeRecyclerView();
                            }
                        }).show();
            }
        });
        initializeRecyclerView();
    }

    public void initializeRecyclerView() {
        SQLiteDatabase database = openOrCreateDatabase("captured_results.db", MODE_PRIVATE, null);
        database.execSQL("CREATE TABLE IF NOT EXISTS results(id_number VARCHAR, name VARCHAR, nationality VARCHAR, gender VARCHAR, birth_year VARCHAR, birth_month VARCHAR, birth_day VARCHAR, address VARCHAR);");
        Cursor resultSet = database.rawQuery("SELECT * FROM results;", null);
        ArrayList<CapturedResultDataObject> arrayList = new ArrayList<>();
        if (resultSet.moveToFirst()) {
            do {
                arrayList.add(new CapturedResultDataObject(
                        resultSet.getString(0),
                        resultSet.getString(1),
                        resultSet.getString(2),
                        resultSet.getString(3),
                        resultSet.getString(4),
                        resultSet.getString(5),
                        resultSet.getString(6),
                        resultSet.getString(7)
                ));
            } while(resultSet.moveToNext());
        }
        resultSet.close();
        database.close();

        LinearLayoutManager layoutManager
                = new LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false);
        binding.recyclerView.setLayoutManager(layoutManager);
        binding.recyclerView.setAdapter(new CapturedResultsAdapter(arrayList));
        DividerItemDecoration dividerItemDecoration = new DividerItemDecoration(binding.recyclerView.getContext(),
                layoutManager.getOrientation());
        binding.recyclerView.addItemDecoration(dividerItemDecoration);
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            NavUtils.navigateUpFromSameTask(this);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public static class CapturedResultDataObject {
        String number;
        String name;
        String nationality;
        String gender;
        String birthYear;
        String birthMonth;
        String birthDay;
        String address;

        public CapturedResultDataObject(String number, String name, String nationality, String gender, String birthYear, String birthMonth, String birthDay, String address) {
            this.name = name;
            this.number = number;
            this.nationality = nationality;
            this.gender = gender;
            this.birthYear = birthYear;
            this.birthMonth = birthMonth;
            this.birthDay = birthDay;
            this.address = address;
        }

        public String getName() {
            return name;
        }

        public String getNumber() {
            return number;
        }

        public String getNationality() {
            return nationality;
        }

        public String getGender() {
            return gender;
        }

        public String getBirthYear() {
            return birthYear;
        }

        public String getBirthMonth() {
            return birthMonth;
        }

        public String getBirthDay() {
            return birthDay;
        }

        public String getAddress() {
            return address;
        }
    }


    public static class CapturedResultsAdapter extends RecyclerView.Adapter<CapturedResultsAdapter.ViewHolder> {

        private final ArrayList<CapturedResultDataObject> localDataSet;

        /**
         * Provide a reference to the type of views that you are using
         * (custom ViewHolder).
         */
        public static class ViewHolder extends RecyclerView.ViewHolder {
            private final TextView numberTextView;
            private final TextView nameTextView;
            private final TextView genderTextView;
            private final TextView nationalityTextView;
            private final TextView birthYearTextView;
            private final TextView birthMonthTextView;
            private final TextView birthDayTextView;
//            private final TextView addressTextView;

            public ViewHolder(View view) {
                super(view);

                numberTextView = view.findViewById(R.id.textview_id_number);
                nameTextView = view.findViewById(R.id.textview_id_name);
                genderTextView = view.findViewById(R.id.textview_id_gender);
                nationalityTextView = view.findViewById(R.id.textview_id_nationality);
                birthYearTextView = view.findViewById(R.id.textview_id_birth_year);
                birthMonthTextView = view.findViewById(R.id.textview_id_birth_month);
                birthDayTextView =  view.findViewById(R.id.textview_id_birth_day);
//                addressTextView = view.findViewById(R.id.textview_id_address);
            }

//            public TextView getAddressTextView() {
//                return addressTextView;
//            }

            public TextView getNumberTextView() {
                return numberTextView;
            }

            public TextView getNameTextView() {
                return nameTextView;
            }

            public TextView getGenderTextView() {
                return genderTextView;
            }

            public TextView getNationalityTextView() {
                return nationalityTextView;
            }

            public TextView getBirthYearTextView() {
                return birthYearTextView;
            }

            public TextView getBirthMonthTextView() {
                return birthMonthTextView;
            }

            public TextView getBirthDayTextView() {
                return birthDayTextView;
            }
        }

        /**
         * Initialize the dataset of the Adapter.
         *
         * @param dataSet String[] containing the data to populate views to be used
         * by RecyclerView.
         */
        public CapturedResultsAdapter(ArrayList<CapturedResultDataObject> dataSet) {
            localDataSet = dataSet;
        }

        // Create new views (invoked by the layout manager)
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {
            // Create a new view, which defines the UI of the list item
            View view = LayoutInflater.from(viewGroup.getContext())
                    .inflate(R.layout.recycler_item, viewGroup, false);

            return new ViewHolder(view);
        }

        // Replace the contents of a view (invoked by the layout manager)
        @Override
        public void onBindViewHolder(ViewHolder viewHolder, final int position) {

            // Get element from your dataset at this position and replace the
            // contents of the view with that element
            viewHolder.getNumberTextView().setText(localDataSet.get(position).getNumber());
            viewHolder.getNameTextView().setText(localDataSet.get(position).getName());
            viewHolder.getGenderTextView().setText(localDataSet.get(position).getGender());
            viewHolder.getNationalityTextView().setText(localDataSet.get(position).getNationality());
            viewHolder.getBirthYearTextView().setText(localDataSet.get(position).getBirthYear());
            viewHolder.getBirthMonthTextView().setText(localDataSet.get(position).getBirthMonth());
            viewHolder.getBirthDayTextView().setText(localDataSet.get(position).getBirthDay());
//            viewHolder.getAddressTextView().setText(localDataSet.get(position).getAddress());
        }

        // Return the size of your dataset (invoked by the layout manager)
        @Override
        public int getItemCount() {
            return localDataSet.size();
        }
    }
}