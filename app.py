import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata, UnivariateSpline
import io
import re

# --- USTAWIENIA STRONY STREAMLIT ---
st.set_page_config(
    page_title="Interaktywny Analizator Żagli 49er / FX",
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
    
    # ZABEZPIECZENIE: Usuwanie duplikatów współrzędnych (x, y) przed interpolacją 2D
    all_points = all_points.drop_duplicates(subset=['distance', 'height'], keep='first')
    
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
    Odporna na małą liczbę punktów pomiarowych u góry żagla.
    """
    results = []
    for height, profile in df_data.iterrows():
        profile_clean = profile.dropna()
        x_measured = profile_clean.index.values.astype(float)
        z_measured = profile_clean.values
        chord_cm = chord_lengths.loc[height]
        
        # ZABEZPIECZENIE: Odrzucenie punktów pomiarowych leżących na lub poza długością cięciwy
        valid_mask = x_measured < chord_cm
        x_measured = x_measured[valid_mask]
        z_measured = z_measured[valid_mask]
        
        x_complete = np.append(x_measured, chord_cm)
        z_complete = np.append(z_measured, 0)
        sort_idx = np.argsort(x_complete)
        
        x_sorted = x_complete[sort_idx]
        z_sorted = z_complete[sort_idx]
        
        # Dynamiczne dopasowanie stopnia krzywej spline do liczby punktów (zapobiega Singular Matrix)
        num_pts = len(x_sorted)
        k_degree = min(3, num_pts - 1)
        
        if k_degree >= 1:
            spline = UnivariateSpline(x_sorted, z_sorted, s=0, k=k_degree)
        else:
            continue
        
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
    df_orig_raw = pd.read_csv(orig_file, sep=';', decimal=',', index_col=0)
    df_mod_raw = pd.read_csv(mod_file, sep=';', decimal=',', index_col=0)
    
    orig_name = orig_file.name.replace('.csv', '')
    mod_name = mod_file.name.replace('.csv', '')
    
    try:
        # Przetwarzanie i normalizacja danych
        df_orig, chords_orig = parse_and_clean_sail(df_orig_raw)
        df_mod, chords_mod = parse_and_clean_sail(df_mod_raw)

        # 1. Obliczenia siatek lokalnych (eliminacja pustych wierszy NaN dla Plotly)
        # Oryginał:
        x_orig_ax = np.arange(0, chords_orig.max() + 5, 5)
        y_orig_ax = np.arange(df_orig.index.min(), df_orig.index.max() + 5, 5)
        X_orig_grid, Y_orig_grid = np.meshgrid(x_orig_ax, y_orig_ax)
        Z_orig = get_smooth_surface_2d(df_orig, chords_orig, X_orig_grid, Y_orig_grid)

        # Modyfikacja:
        x_mod_ax = np.arange(0, chords_mod.max() + 5, 5)
        y_mod_ax = np.arange(df_mod.index.min(), df_mod.index.max() + 5, 5)
        X_mod_grid, Y_mod_grid = np.meshgrid(x_mod_ax, y_mod_ax)
        Z_mod = get_smooth_surface_2d(df_mod, chords_mod, X_mod_grid, Y_mod_grid)

        # 2. Obliczenie siatki wspólnej (tylko dla części pokrywającej się - bezpieczna dla wykresu różnic)
        common_max_chord = min(chords_orig.max(), chords_mod.max())
        common_max_height = min(df_orig.index.max(), df_mod.index.max())
        common_min_height = max(df_orig.index.min(), df_mod.index.min())

        x_comm_ax = np.arange(0, common_max_chord + 5, 5)
        y_comm_ax = np.arange(common_min_height, common_max_height + 5, 5)
        X_comm, Y_comm = np.meshgrid(x_comm_ax, y_comm_ax)

        # Wygładzenie na siatce wspólnej do celów porównawczych
        Z_orig_comm = get_smooth_surface_2d(df_orig, chords_orig, X_comm, Y_comm)
        Z_mod_comm = get_smooth_surface_2d(df_mod, chords_mod, X_comm, Y_comm)
        
        Z_diff = Z_mod_comm - Z_orig_comm
        max_abs_diff = np.nanmax(np.abs(Z_diff))
        global_max_depth = np.nanmax([np.nanmax(Z_orig), np.nanmax(Z_mod)])

        # Obliczenia tabelaryczne
        table_orig = analyze_profile_geometry(df_orig, chords_orig)
        table_mod = analyze_profile_geometry(df_mod, chords_mod)

        # --- ZAKŁADKI W INTERFEJSIE ---
        tab1, tab2, tab3 = st.tabs(["📊 Porównanie 3D", "🔍 Wykres Różnicowy 3D", "📋 Parametry & Raport Excel"])

        with tab1:
            st.header("Interaktywne porównanie geometrii żagli (Obrotowe modele 3D)")
            st.write("Użyj myszki, aby obracać, przybliżać (scroll) i przesuwać wykresy.")
            
            col1, col2 = st.columns(2)
            
            # Scena dla oryginału
            scene_orig = dict(
                aspectratio=dict(x=1, y=df_orig.index.max()/chords_orig.max(), z=(global_max_depth/10/chords_orig.max())*2),
                xaxis=dict(title='Odległość (cm)'),
                yaxis=dict(title='Wysokość (cm)'),
                zaxis=dict(title='Głębokość (mm)', range=[0, global_max_depth])
            )
            # Scena dla modyfikacji
            scene_mod = dict(
                aspectratio=dict(x=1, y=df_mod.index.max()/chords_mod.max(), z=(global_max_depth/10/chords_mod.max())*2),
                xaxis=dict(title='Odległość (cm)'),
                yaxis=dict(title='Wysokość (cm)'),
                zaxis=dict(title='Głębokość (mm)', range=[0, global_max_depth])
            )

            with col1:
                st.subheader(f"Oryginał: {orig_name}")
                fig1 = go.Figure(data=[go.Surface(x=x_orig_ax, y=y_orig_ax, z=Z_orig, colorscale='viridis', cmin=0, cmax=global_max_depth)])
                fig1.update_layout(scene=scene_orig, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader(f"Modyfikacja: {mod_name}")
                fig2 = go.Figure(data=[go.Surface(x=x_mod_ax, y=y_mod_ax, z=Z_mod, colorscale='viridis', cmin=0, cmax=global_max_depth)])
                fig2.update_layout(scene=scene_mod, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.header("Interaktywny Wykres Różnicowy 3D")
            st.write("Czerwony kolor = żagiel zmodyfikowany jest głębszy. Niebieski = żagiel oryginalny jest głębszy (modyfikacja spłaszczona).")
            
            fig_diff = go.Figure()
            
            # Oryginał jako półprzezroczysty szary punkt odniesienia
            fig_diff.add_trace(go.Surface(
                x=x_comm_ax, y=y_comm_ax, z=Z_orig_comm,
                colorscale=[[0, 'grey'], [1, 'grey']],
                showscale=False,
                opacity=0.15,
                hoverinfo='skip'
            ))
            
            # Powierzchnia różnicowa (pokolorowana przez Z_diff na siatce wspólnej)
            fig_diff.add_trace(go.Surface(
                x=x_comm_ax, y=y_comm_ax, z=Z_mod_comm,
                surfacecolor=Z_diff,
                colorscale='rdbu',
                cmin=-max_abs_diff,
                cmax=max_abs_diff,
                colorbar=dict(title="Różnica (mm)")
            ))
            
            scene_diff = dict(
                aspectratio=dict(x=1, y=common_max_height/common_max_chord, z=(global_max_depth/10/common_max_chord)*2),
                xaxis=dict(title='Odległość (cm)'),
                yaxis=dict(title='Wysokość (cm)'),
                zaxis=dict(title='Różnica (mm)')
            )
            fig_diff.update_layout(scene=scene_diff, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_diff, use_container_width=True)

        with tab3:
            st.header("Analiza Parametryczna Profili")
            
            # Generowanie skoroszytu Excel w pamięci RAM serwera
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
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
    st.info("👈 Aby rozpocząć analizę, prześlij oba pliki CSV (Oryginalny oraz Zmodyfikowany) w panelu bocznym po lewej stronie.")