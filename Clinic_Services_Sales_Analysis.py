import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# Γεννήτρια τυχαίων δεδομένων με μεγαλύτερη διαφοροποίηση
np.random.seed(42)
num_rows = 100

dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
services = ["Ακτινολογία", "Φυσιοθεραπεία", "Αναλύσεις Αίματος", "Αναλύσεις Ούρων", "Γενική Ιατρική", "Παιδιατρική", "Οδοντιατρική"]
names = ["Γιάννης", "Μαρία", "Κώστας", "Ελένη", "Νίκος", "Σοφία", "Αλέξανδρος", "Κατερίνα"]
zip_codes = ["11145", "14562", "18974", "10245", "19876"]

# Δημιουργία διαφοροποιημένων ηλικιών και αξίας υπηρεσίας
ages = np.random.randint(18, 80, num_rows)

# Δημιουργία πιο διαφοροποιημένων αξιών υπηρεσιών
service_values = np.random.randint(50, 500, num_rows)
service_values[:25] += 200  # Υψηλές αξίες για τις πρώτες 25 γραμμές (π.χ., ακριβές υπηρεσίες)
service_values[25:50] -= 100  # Χαμηλές αξίες για τις επόμενες 25 γραμμές
service_values[50:] += 100  # Μέτριες αξίες για τις επόμενες γραμμές

data = {
    "Όνομα Πελάτη": np.random.choice(names, num_rows),
    "Ημερομηνία": np.random.choice(dates, num_rows),
    "Υπηρεσία": np.random.choice(services, num_rows),
    "Αξία Υπηρεσίας": service_values,
    "Ηλικία": ages,
    "Ταχυδρομικός Κώδικας": np.random.choice(zip_codes, num_rows)
}

df = pd.DataFrame(data)
df.sort_values("Ημερομηνία", inplace=True)
df.reset_index(drop=True, inplace=True)

# Streamlit app
st.set_page_config(layout="wide")
st.title("📊 Ανάλυση Πωλήσεων Υπηρεσιών Πολυιατρείου")

# Sidebar φίλτρα
st.sidebar.header("🔍 Φίλτρα")
selected_service = st.sidebar.multiselect("Επιλέξτε Υπηρεσίες", options=df["Υπηρεσία"].unique(), default=df["Υπηρεσία"].unique())
selected_zip = st.sidebar.multiselect("Επιλέξτε Ταχυδρομικό Κώδικα", options=df["Ταχυδρομικός Κώδικας"].unique(), default=df["Ταχυδρομικός Κώδικας"].unique())
selected_age_range = st.sidebar.slider("Εύρος Ηλικίας", min_value=df["Ηλικία"].min(), max_value=df["Ηλικία"].max(), value=(df["Ηλικία"].min(), df["Ηλικία"].max()))

# Φιλτράρισμα δεδομένων
filtered_df = df[(df["Υπηρεσία"].isin(selected_service)) &
                 (df["Ταχυδρομικός Κώδικας"].isin(selected_zip)) &
                 (df["Ηλικία"].between(selected_age_range[0], selected_age_range[1]))]

# Εμφάνιση δεδομένων
st.write("### 🔢 Πίνακας Δεδομένων")
st.dataframe(filtered_df)

# Συνολικά έσοδα και ασθενείς
total_revenue = filtered_df["Αξία Υπηρεσίας"].sum()
total_patients = len(filtered_df)

col1, col2 = st.columns(2)
col1.metric(label="💰 Συνολικά Έσοδα", value=f"€{total_revenue}")
col2.metric(label="👥 Συνολικοί Πελάτες", value=total_patients)

# Υπολογισμός συνολικής δαπάνης ανά πελάτη
total_spend_per_patient = filtered_df.groupby("Όνομα Πελάτη")["Αξία Υπηρεσίας"].sum().reset_index()

# Εμφάνιση των δεδομένων με τη συνολική δαπάνη
st.write("### 📊 Συνολική Δαπάνη Ανά Ασθενή")
st.dataframe(total_spend_per_patient)



# Ανάλυση κατά υπηρεσία
st.write("### 📈 Ανάλυση Εσόδων ανά Υπηρεσία")
service_summary = filtered_df.groupby("Υπηρεσία")["Αξία Υπηρεσίας"].sum().reset_index()
fig_service = px.bar(service_summary, x="Υπηρεσία", y="Αξία Υπηρεσίας", title="Ανάλυση Εσόδων ανά Υπηρεσία")
st.plotly_chart(fig_service)

# Οπτικοποίηση κατανομής ηλικιών (Bar plot)
st.write("### 📊 Κατανομή Ηλικιών (Bar plot)")
fig_age_bar = px.histogram(filtered_df, x="Ηλικία", nbins=10, title="Κατανομή Ηλικιών (Bar plot)")
st.plotly_chart(fig_age_bar)

# Οπτικοποίηση κατανομής ηλικιών (Box plot)
st.write("### 📊 Κατανομή Ηλικιών (Box plot)")
fig_age_box = px.box(filtered_df, y="Ηλικία", title="Κατανομή Ηλικιών (Box plot)")
st.plotly_chart(fig_age_box)

# K-Means Clustering
st.write("### 🔍 Ανάλυση Κλαστικοποίησης με K-Means")
X = filtered_df[["Αξία Υπηρεσίας", "Ηλικία"]]
kmeans = KMeans(n_clusters=4, random_state=42)
filtered_df["Ομάδα"] = kmeans.fit_predict(X)

fig_cluster = px.scatter(filtered_df, x="Αξία Υπηρεσίας", y="Ηλικία", color="Ομάδα", title="Οπτικοποίηση Κλασμάτων (Clusters)")
st.plotly_chart(fig_cluster)

# Ανάλυση των ομάδων
cluster_summary = filtered_df.groupby("Ομάδα")[["Αξία Υπηρεσίας", "Ηλικία"]].mean().reset_index()
st.write("### 📊 Ανάλυση Ομάδων Κλαστικοποίησης")
st.dataframe(cluster_summary)

# Σχόλιο για την ανάλυση
st.write("""
**Σχόλιο για την Ανάλυση Ομάδων:**

Η ανάλυση K-Means που παρουσιάζεται παρακάτω κατατάσσει τους πελάτες σε τέσσερις ομάδες, με βάση τη δαπάνη τους και την ηλικία τους. Αυτές οι ομάδες είναι υποθετικές και ενδεικτικές, καθώς οι πραγματικές ομάδες θα διαμορφωθούν από τα δεδομένα που εισάγονται στην εφαρμογή. Οι ομάδες που παρουσιάζονται μπορεί να διαφέρουν ανάλογα με τα δεδομένα και τις παραμέτρους που χρησιμοποιούνται στην ανάλυση. 

Ωστόσο, τα αποτελέσματα αυτής της ανάλυσης παρέχουν πιθανές κατηγορίες πελατών που μπορούν να αξιοποιηθούν στρατηγικά. Κάθε ομάδα μπορεί να εξεταστεί περαιτέρω για να αναπτυχθούν στρατηγικές προσαρμοσμένες στις ανάγκες της:

- **Ομάδα 1 (Υψηλή Δαπάνη, Νεότεροι Πελάτες)**: Αυτή η ομάδα περιλαμβάνει πελάτες που πληρώνουν υψηλότερες αξίες και είναι νεότεροι σε ηλικία. Υποθετικά, αυτοί οι πελάτες μπορεί να ενδιαφέρονται για νέες, καινοτόμες υπηρεσίες και προϊόντα.

- **Ομάδα 2 (Χαμηλή Δαπάνη, Νεότεροι Πελάτες)**: Οι πελάτες αυτής της ομάδας πληρώνουν χαμηλότερες αξίες και είναι νεότεροι σε ηλικία. Εδώ, ίσως μπορούμε να προσφέρουμε οικονομικές υπηρεσίες ή στοχευμένα προϊόντα που ανταποκρίνονται στις ανάγκες τους.

- **Ομάδα 3 (Μεσαία Δαπάνη, Μεγαλύτεροι Πελάτες)**: Αυτή η ομάδα περιλαμβάνει μεγαλύτερους πελάτες που πληρώνουν μέτριες αξίες. Θα μπορούσαμε να προσανατολιστούμε σε υπηρεσίες ή προϊόντα που τους εξυπηρετούν καλύτερα, π.χ. γενικές ιατρικές υπηρεσίες ή υπηρεσίες για οικογένειες.

- **Ομάδα 4 (Υψηλή Δαπάνη, Μεγαλύτεροι Πελάτες)**: Αυτή η ομάδα περιλαμβάνει πελάτες που πληρώνουν υψηλές αξίες και είναι μεγαλύτερης ηλικίας. Ίσως είναι πιο ενδιαφέρον για προϊόντα ή υπηρεσίες υψηλής ποιότητας ή για εξειδικευμένα προϊόντα που ανταποκρίνονται στις ειδικές ανάγκες τους.
""")
