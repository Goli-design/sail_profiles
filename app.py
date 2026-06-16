import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, UnivariateSpline
import plotly.graph_objects as go
import io
import re

# --- USTAWIENIA STRONY STREAMLIT ---
st.set_page_config(
    page_title="Analizator Profili Żagli 49er / FX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNKCJE POMOCNICZE (MATEMATYKA I PRZETWARZANIE) ---

def parse_and_clean_sail(df_full):
    """
    Oczyszcza dane wejściowe, wyodrębnia długości cięciw i ujednolica jednostki do [cm].
    """
    chord_col = next((col for col in df_full.columns if 'chord' in col.lower()), None)
    if not chord_col:
        raise ValueError("Plik CSV musi zawierać kolumnę z długością cięciwy (np. 'Chord length').")
        
    chord_lengths = df_full[chord_col].copy()
    # Detekcja mm i konwersja na cm
    if chord_lengths.max() > 300:
        chord_lengths = chord_lengths / 10.0
        
    df_data = df_full.drop(columns=[chord_col])
    df_data.columns = pd.to_numeric(df_data.columns)
    
    return df_data, chord_lengths

def get_smooth_surface_2d(df_data, chord_lengths, grid_x, grid_y):
    """
    Tworzy wygładzoną powierzchnię 2D żagla za pomocą interpolacji sześciennej griddata.
    """
    leech_points = pd.DataFrame({'height': chord_lengths.index, 'distance': chord_lengths.values, 'depth': 0})
    df_stacked = df_data.stack().reset_index()
    df_stacked.columns = ['height', 'distance', 'depth']
    
    all_points = pd.concat([df_stacked, leech_points], ignore_index=True)
    points = all_points[['distance', 'height']].values
    values = all_points['depth'].values

    Z_grid = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    # Przycinanie krawędzi liku wolnego
    for i, y_val in enumerate(grid_y[:, 0]):
        closest_y_idx = np.abs(df_data.index.to_numpy() - y_val).argmin()
        closest_y = df_data.index[closest_y_idx]
        max_x = chord_lengths.loc[closest_y]
        if max_x is not None:
            Z_grid[i, grid_x[i, :] > max_x] = np.nan
            
    Z_grid[:, 0] = 0
    return Z_grid

def analyze_profile_geometry(df_data, chord_lengths):
    """
    Oblicza 8 parametrów aerodynamicznych profilu dla każdej wysokości żagla.
    """
    results = []
    for height, profile in df_data.iterrows():
        profile_clean = profile.dropna()
        x_measured = profile_clean.index.values.astype(float)
        z_measured = profile_clean.values
        chord_cm = chord_lengths.loc[height]
        
        x_complete = np.append(x_measured, chord_cm)
        z_complete = np.append(z_measured, 0)
        sort_idx = np.argsort(x_complete)
        
        spline = UnivariateSpline(x_complete[sort_idx], z_complete[sort_idx], s=0, k=3)
        
        x_fine = np.linspace(0, chord_cm, 2000)
        z_fine = spline(x_fine)
        
        max_depth_mm = np.max(z_fine)
        max_depth_pos_cm = x_fine[np.argmax(z_fine)]

        if chord_cm > 0:
            max_depth_perc_chord = (max_depth_mm / 10 / chord_cm) * 100
            max_depth_pos_perc_chord = (max_depth_pos_cm / chord_cm) * 100
        else:
            max_depth_perc_chord = max_depth_pos_perc_chord = 0
        
        x_front_mid = max_depth_pos_cm / 2
        front_depth_mm = spline(x_front_mid)

        x_rear_mid = max_depth_pos_cm + (chord_cm - max_depth_pos_cm) / 2
        rear_depth_mm = spline(x_rear_mid)

        if max_depth_mm > 0:
            front_depth_perc_max = (front_depth_mm / max_depth_mm) * 100
            rear_depth_perc_max = (rear_depth_mm / max_depth_mm) * 100
        else:
            front_depth_perc_max = rear_depth_perc_max = 0
        
        spline_deriv = spline.derivative(n=1)
        slope_entry = spline_deriv(0) / 10.0
        entry_angle_deg = np.degrees(np.arctan(slope_entry))

        slope_exit = spline_deriv(chord_cm) / 10.0
        exit_angle_deg = np.degrees(np.arctan(slope_exit))

        results.append({
            'Wysokość (cm)': height,
            'Maks. głębokość (% cięciwy)': round(max_depth_perc_chord, 1),
            'Poz. maks. głębokości (% cięciwy)': round(max_depth_pos_perc_chord, 1),
            'Głęb. przednia (% maks.)': round(front_depth_perc_max, 1),
            'Głęb. tylna (% maks.)': round(rear_depth_perc_max, 1),
            'Kąt natarcia (stopnie)': round(entry_angle_deg, 1),
            'Kąt spływu (stopnie)': round(exit_angle_deg, 1)
        })
        
    return pd.DataFrame(results).set_index('Wysokość (cm)')

# --- INTERFEJS UŻYTKOWNIKA ---

st.title("⛵ Aerodynamiczny Analizator i Komparator Żagli")
st.markdown("Narzędzie dedykowane dla klas **49er** oraz **49er FX**. Porównuje dwa projekty żagli w przestrzeni 3D oraz oblicza parametry profili.")

# Panel boczny - Przesyłanie plików
st.sidebar.header("📁 Wczytywanie danych")
orig_file = st.sidebar.file_uploader("Wybierz żagiel ORYGINALNY (CSV)", type="csv")
mod_file = st.sidebar.file_uploader("Wybierz żagiel ZMODYFIKOWANY (CSV)", type="csv")

if orig_file and mod_file:
    # Wczytanie plików wejściowych
    df_orig_raw = pd.read_csv(orig_file, sep=';', decimal=',', index_col=0)
    df_mod_raw = pd.read_csv(mod_file, sep=';', decimal=',', index_col=0)
    
    orig_name = orig_file.name.replace('.csv', '')
    mod_name = mod_file.name.replace('.csv', '')
    
    try:
        # Przetwarzanie i normalizacja danych
        df_orig, chords_orig = parse_and_clean_sail(df_orig_raw)
        df_mod, chords_mod = parse_and_clean_sail(df_mod_raw)
        
        # Tworzenie wspólnej Siatki Głównej do interpolacji
        max_chord = max(chords_orig.max(), chords_mod.max())
        max_height = max(df_orig.index.max(), df_mod.index.max())
        min_height = min(df_orig.index.min(), df_mod.index.min())

        x_master = np.arange(0, max_chord + 5, 5)
        y_master = np.arange(min_height, max_height + 5, 5)
        X_master, Y_master = np.meshgrid(x_master, y_master)

        # Wygładzanie 2D powierzchni żagli
        Z_orig = get_smooth_surface_2d(df_orig, chords_orig, X_master, Y_master)
        Z_mod = get_smooth_surface_2d(df_mod, chords_mod, X_master, Y_master)
        
        global_max_depth = np.nanmax([Z_orig, Z_mod])
        Z_diff = Z_mod - Z_orig
        max_abs_diff = np.nanmax(np.abs(Z_diff))

        # Obliczenia parametrów 2D
        table_orig = analyze_profile_geometry(df_orig, chords_orig)
        table_mod = analyze_profile_geometry(df_mod, chords_mod)

        # --- ZAKŁADKI W INTERFEJSIE ---
        tab1, tab2, tab3 = st.tabs(["📊 Porównanie 3D", "🔍 Wykres Różnicowy 3D", "📋 Parametry & Raport Excel"])

        with tab1:
            st.header("Porównanie geometrii żagli (Wygładzone modele 3D)")
            st.write("Wykresy są interaktywne. Możesz je obracać myszką, przybliżać i sprawdzać wartości punktów.")
            
            col1, col2 = st.columns(2)
            
            # Parametry sceny Plotly zachowujące realne proporcje z 2x przewyższeniem osi Z
            scene_layout = dict(
                aspectratio=dict(x=1, y=max_height/max_chord, z=(global_max_depth/10/max_chord)*2),
                xaxis=dict(title='Odległość (cm)'),
                yaxis=dict(title='Wysokość (cm)'),
                zaxis=dict(title='Głębokość (mm)', range=[0, global_max_depth])
            )

            with col1:
                st.subheader(f"Oryginał: {orig_name}")
                fig1 = go.Figure(data=[go.Surface(x=X_master, y=Y_master, z=Z_orig, colorscale='Viridis', cmin=0, cmax=global_max_depth)])
                fig1.update_layout(scene=scene_layout, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader(f"Modyfikacja: {mod_name}")
                fig2 = go.Figure(data=[go.Surface(x=X_master, y=Y_master, z=Z_mod, colorscale='Viridis', cmin=0, cmax=global_max_depth)])
                fig2.update_layout(scene=scene_layout, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.header("Trójwymiarowy Wykres Różnicowy")
            st.write("Czerwony kolor oznacza miejsca, gdzie żagiel zmodyfikowany jest głębszy. Niebieski - gdzie jest płaski.")
            
            fig_diff = go.Figure()
            
            # Powierzchnia oryginalnego żagla jako półprzezroczyste szare odniesienie
            fig_diff.add_trace(go.Surface(
                x=X_master, y=Y_master, z=Z_orig,
                colorscale=[[0, 'grey'], [1, 'grey']],
                showscale=False,
                opacity=0.2,
                hoverinfo='skip'
            ))
            
            # Powierzchnia zmodyfikowana pokolorowana wartościami różnic (coolwarm)
            fig_diff.add_trace(go.Surface(
                x=X_master, y=Y_master, z=Z_mod,
                surfacecolor=Z_diff,
                colorscale='Coolwarm',
                cmin=-max_abs_diff,
                cmax=max_abs_diff,
                colorbar=dict(title="Różnica (mm)")
            ))
            
            scene_layout_diff = scene_layout.copy()
            scene_layout_diff['zaxis'] = dict(title='Głębokość (mm)')
            fig_diff.update_layout(scene=scene_layout_diff, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_diff, use_container_width=True)

        with tab3:
            st.header("Analiza Parametryczna Profili")
            
            # Generowanie skoroszytu Excel w pamięci RAM serwera
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Oczyszczanie i skracanie nazw arkuszy do limitu Excela (30 znaków)
                sheet_orig = re.sub(r'[\\/*?:\[\]]', '', orig_name)[:30]
                sheet_mod = re.sub(r'[\\/*?:\[\]]', '', mod_name)[:30]
                
                table_orig.to_excel(writer, sheet_name=sheet_orig)
                table_mod.to_excel(writer, sheet_name=sheet_mod)
                
            st.download_button(
                label="📥 Pobierz wyniki w jednym pliku Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"analiza_porownawcza_{orig_name}_vs_{mod_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.subheader(f"Oryginał: {orig_name}")
                st.dataframe(table_orig)
                
            with col_t2:
                st.subheader(f"Modyfikacja: {mod_name}")
                st.dataframe(table_mod)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania plików. Upewnij się, że oba pliki posiadają prawidłową strukturę. Szczegóły błędu: {e}")

else:
    # Komunikat startowy, gdy pliki nie zostały jeszcze wczytane
    st.info("👈 Aby rozpocząć analizę, prześlij oba pliki CSV (Oryginalny oraz Zmodyfikowany) w panelu bocznym po lewej stronie.")