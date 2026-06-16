import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata, UnivariateSpline
import io
import re

# --- USTAWIENIA STRONY STREAMLIT ---
st.set_page_config(
    page_title="Interaktywny Analizator Żagli 49er / FX (mm)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNKCJE POMOCNICZE (CAŁOŚĆ W MILIMETRACH [mm]) ---

def parse_and_clean_sail_mm(df_full):
    """
    Wczytuje dane żagla, gdzie wszystkie wartości (wysokość, odległość, głębokość, cięciwa) są w [mm].
    """
    chord_col = next((col for col in df_full.columns if 'chord' in col.lower()), None)
    if not chord_col:
        raise ValueError("Plik CSV musi zawierać kolumnę z długością cięciwy (np. 'Chord length').")
        
    chord_lengths = df_full[chord_col].copy()
    df_data = df_full.drop(columns=[chord_col])
    df_data.columns = pd.to_numeric(df_data.columns)
    
    return df_data, chord_lengths

def get_smooth_surface_2d_mm(df_data, chord_lengths, grid_x, grid_y):
    """
    Tworzy wygładzoną powierzchnię 2D żagla za pomocą interpolacji sześciennej griddata.
    """
    leech_points = pd.DataFrame({'height': chord_lengths.index, 'distance': chord_lengths.values, 'depth': 0})
    df_stacked = df_data.stack().reset_index()
    df_stacked.columns = ['height', 'distance', 'depth']
    
    all_points = pd.concat([df_stacked, leech_points], ignore_index=True)
    
    # ZABEZPIECZENIE: Usuwanie duplikatów współrzędnych przed interpolacją 2D
    all_points = all_points.drop_duplicates(subset=['distance', 'height'], keep='first')
    
    points = all_points[['distance', 'height']].values
    values = all_points['depth'].values

    Z_grid = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # <<< ROZWIĄZANIE: Płynna interpolacja liku wolnego (eliminacja schodków) >>>
    # Zamiast szukać najbliższego profilu, obliczamy dokładną, płynną cięciwę dla każdej wysokości siatki
    for i, y_val in enumerate(grid_y[:, 0]):
        max_x = np.interp(y_val, chord_lengths.index, chord_lengths.values)
        Z_grid[i, grid_x[i, :] > max_x] = np.nan
            
    Z_grid[:, 0] = 0
    return Z_grid

def analyze_profile_geometry_mm(df_data, chord_lengths):
    """
    Oblicza 8 parametrów aerodynamicznych profilu dla każdej wysokości żagla.
    Wszystkie obliczenia i dane wejściowe/wyjściowe są w [mm].
    """
    results = []
    for height_mm, profile in df_data.iterrows():
        profile_clean = profile.dropna()
        x_measured = profile_clean.index.values.astype(float)
        z_measured = profile_clean.values
        chord_mm = chord_lengths.loc[height_mm]
        
        # ZABEZPIECZENIE: Odrzucenie punktów leżących poza długością cięciwy
        valid_mask = x_measured < chord_mm
        x_measured = x_measured[valid_mask]
        z_measured = z_measured[valid_mask]
        
        x_complete = np.append(x_measured, chord_mm)
        z_complete = np.append(z_measured, 0)
        sort_idx = np.argsort(x_complete)
        
        x_sorted = x_complete[sort_idx]
        z_sorted = z_complete[sort_idx]
        
        # Dynamiczne dopasowanie stopnia krzywej spline do liczby punktów
        num_pts = len(x_sorted)
        k_degree = min(3, num_pts - 1)
        
        if k_degree >= 1:
            spline = UnivariateSpline(x_sorted, z_sorted, s=0, k=k_degree)
        else:
            continue
        
        x_fine = np.linspace(0, chord_mm, 2000)
        z_fine = spline(x_fine)
        
        max_depth_mm = np.max(z_fine)
        max_depth_pos_mm = x_fine[np.argmax(z_fine)]

        if chord_mm > 0:
            max_depth_perc_chord = (max_depth_mm / chord_mm) * 100
            max_depth_pos_perc_chord = (max_depth_pos_mm / chord_mm) * 100
        else:
            max_depth_perc_chord = max_depth_pos_perc_chord = 0
        
        x_front_mid = max_depth_pos_mm / 2
        front_depth_mm = spline(x_front_mid)

        x_rear_mid = max_depth_pos_mm + (chord_mm - max_depth_pos_mm) / 2
        rear_depth_mm = spline(x_rear_mid)

        if max_depth_mm > 0:
            front_depth_perc_max = (front_depth_mm / max_depth_mm) * 100
            rear_depth_perc_max = (rear_depth_mm / max_depth_mm) * 100
        else:
            front_depth_perc_max = rear_depth_perc_max = 0
        
        spline_deriv = spline.derivative(n=1)
        slope_entry = spline_deriv(0)
        entry_angle_deg = np.degrees(np.arctan(slope_entry))

        slope_exit = spline_deriv(chord_mm)
        exit_angle_deg = np.degrees(np.arctan(slope_exit))

        results.append({
            'Wysokość (mm)': height_mm,
            'Maks. głębokość (% cięciwy)': round(max_depth_perc_chord, 1),
            'Poz. maks. głębokości (% cięciwy)': round(max_depth_pos_perc_chord, 1),
            'Głęb. przednia (% maks.)': round(front_depth_perc_max, 1),
            'Głęb. tylna (% maks.)': round(rear_depth_perc_max, 1),
            'Kąt natarcia (stopnie)': round(entry_angle_deg, 1),
            'Kąt spływu (stopnie)': round(exit_angle_deg, 1)
        })
        
    return pd.DataFrame(results).set_index('Wysokość (mm)')

# --- INTERFEJS UŻYTKOWNIKA ---

st.title("⛵ Aerodynamiczny Analizator i Komparator Żagli [Skala mm]")
st.markdown("Narzędzie obsługuje pliki pomiarowe, w których **wszystkie wymiary są wyrażone w milimetrach [mm]**.")

# Panel boczny - Przesyłanie plików
st.sidebar.header("📁 Wczytywanie danych")
orig_file = st.sidebar.file_uploader("Wybierz żagiel ORYGINALNY (CSV w mm)", type="csv")
mod_file = st.sidebar.file_uploader("Wybierz żagiel ZMODYFIKOWANY (CSV w mm)", type="csv")

if orig_file and mod_file:
    # Wczytanie plików wejściowych
    df_orig_raw = pd.read_csv(orig_file, sep=';', decimal=',', index_col=0)
    df_mod_raw = pd.read_csv(mod_file, sep=';', decimal=',', index_col=0)
    
    orig_name = orig_file.name.replace('.csv', '')
    mod_name = mod_file.name.replace('.csv', '')
    
    try:
        # Przetwarzanie danych
        df_orig, chords_orig = parse_and_clean_sail_mm(df_orig_raw)
        df_mod, chords_mod = parse_and_clean_sail_mm(df_mod_raw)
        
        # Obliczenie maksymalnych wymiarów siatki głównej (w całości w [mm])
        max_chord = max(chords_orig.max(), chords_mod.max())
        max_height = max(df_orig.index.max(), df_mod.index.max())
        min_height = min(df_orig.index.min(), df_mod.index.min())

        # Siatka obliczeniowa z krokiem co 50 mm (czyli 5 cm)
        x_orig_ax = np.arange(0, chords_orig.max() + 50, 50)
        y_orig_ax = np.arange(df_orig.index.min(), df_orig.index.max() + 50, 50)
        X_orig_grid, Y_orig_grid = np.meshgrid(x_orig_ax, y_orig_ax)
        Z_orig = get_smooth_surface_2d_mm(df_orig, chords_orig, X_orig_grid, Y_orig_grid)

        x_mod_ax = np.arange(0, chords_mod.max() + 50, 50)
        y_mod_ax = np.arange(df_mod.index.min(), df_mod.index.max() + 50, 50)
        X_mod_grid, Y_mod_grid = np.meshgrid(x_mod_ax, y_mod_ax)
        Z_mod = get_smooth_surface_2d_mm(df_mod, chords_mod, X_mod_grid, Y_mod_grid)

        # Obliczenie siatki wspólnej dla wykresu różnic
        common_max_chord = min(chords_orig.max(), chords_mod.max())
        common_max_height = min(df_orig.index.max(), df_mod.index.max())
        common_min_height = max(df_orig.index.min(), df_mod.index.min())

        x_comm_ax = np.arange(0, common_max_chord + 50, 50)
        y_comm_ax = np.arange(common_min_height, common_max_height + 50, 50)
        X_comm, Y_comm = np.meshgrid(x_comm_ax, y_comm_ax)

        # Wygładzenie na siatce wspólnej
        Z_orig_comm = get_smooth_surface_2d_mm(df_orig, chords_orig, X_comm, Y_comm)
        Z_mod_comm = get_smooth_surface_2d_mm(df_mod, chords_mod, X_comm, Y_comm)
        
        Z_diff = Z_mod_comm - Z_orig_comm
        max_abs_diff = np.nanmax(np.abs(Z_diff))
        global_max_depth = np.nanmax([np.nanmax(Z_orig), np.nanmax(Z_mod)])

        # Obliczenia parametrów 2D
        table_orig = analyze_profile_geometry_mm(df_orig, chords_orig)
        table_mod = analyze_profile_geometry_mm(df_mod, chords_mod)

        # --- ZAKŁADKI W INTERFEJSIE ---
        tab1, tab2, tab3 = st.tabs(["📊 Porównanie 3D [mm]", "🔍 Wykres Różnicowy 3D [mm]", "📋 Parametry & Raport Excel"])

        # <<< ROZWIĄZANIE: Precyzyjny dobór proporcji osi (aspectratio) na podstawie rzeczywistych wymiarów [mm] >>>
        # Zapewnia naturalny wygląd żagla i dokładnie dwukrotne (2x) powiększenie osi Z.
        y_to_x_ratio_orig = (df_orig.index.max() - df_orig.index.min()) / chords_orig.max()
        z_to_x_ratio_orig = (global_max_depth / chords_orig.max()) * 2.0  # Dokładnie 2-krotne powiększenie osi Z
        
        y_to_x_ratio_mod = (df_mod.index.max() - df_mod.index.min()) / chords_mod.max()
        z_to_x_ratio_mod = (global_max_depth / chords_mod.max()) * 2.0

        scene_orig = dict(
            aspectratio=dict(x=1.0, y=y_to_x_ratio_orig, z=z_to_x_ratio_orig),
            xaxis=dict(title='Odległość (mm)'),
            yaxis=dict(title='Wysokość (mm)'),
            zaxis=dict(title='Głębokość (mm)', range=[0, global_max_depth])
        )
        scene_mod = dict(
            aspectratio=dict(x=1.0, y=y_to_x_ratio_mod, z=z_to_x_ratio_mod),
            xaxis=dict(title='Odległość (mm)'),
            yaxis=dict(title='Wysokość (mm)'),
            zaxis=dict(title='Głębokość (mm)', range=[0, global_max_depth])
        )

        with tab1:
            st.header("Interaktywne porównanie geometrii żagli (Obrotowe modele 3D)")
            st.write("Użyj myszki, aby obracać, przybliżać (scroll) i przesuwać wykresy. Wszystkie osie w [mm].")
            
            col1, col2 = st.columns(2)

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
            st.write("Czerwony kolor = żagiel zmodyfikowany jest głębszy. Niebieski = spłaszczony. Wszystkie osie w [mm].")
            
            fig_diff = go.Figure()
            
            # Oryginał jako półprzezroczysty szary punkt odniesienia
            fig_diff.add_trace(go.Surface(
                x=x_comm_ax, y=y_comm_ax, z=Z_orig_comm,
                colorscale=[[0, 'grey'], [1, 'grey']],
                showscale=False,
                opacity=0.15,
                hoverinfo='skip'
            ))
            
            # Powierzchnia różnicowa
            fig_diff.add_trace(go.Surface(
                x=x_comm_ax, y=y_comm_ax, z=Z_mod_comm,
                surfacecolor=Z_diff,
                colorscale='rdbu',
                cmin=-max_abs_diff,
                cmax=max_abs_diff,
                colorbar=dict(title="Różnica (mm)")
            ))
            
            # Skalowanie różnicy na siatce wspólnej
            y_to_x_ratio_comm = (common_max_height - common_min_height) / common_max_chord
            z_to_x_ratio_comm = (global_max_depth / common_max_chord) * 2.0
            
            scene_diff = dict(
                aspectratio=dict(x=1.0, y=y_to_x_ratio_comm, z=z_to_x_ratio_comm),
                xaxis=dict(title='Odległość (mm)'),
                yaxis=dict(title='Wysokość (mm)'),
                zaxis=dict(title='Różnica (mm)')
            )
            fig_diff.update_layout(scene=scene_diff, margin=dict(l=0, r=0, b=0, t=0))
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
                
            with st.expander("🔍 Diagnostyka danych (Debug)"):
                st.write("**Oryginał:**")
                st.write(f"- Min głębokość [mm]: {np.nanmin(Z_orig):.1f}")
                st.write(f"- Max głębokość [mm]: {np.nanmax(Z_orig):.1f}")
                st.write(f"- Liczba poprawnych punktów siatki: {np.count_nonzero(~np.isnan(Z_orig))}")
                st.write("**Modyfikacja:**")
                st.write(f"- Min głębokość [mm]: {np.nanmin(Z_mod):.1f}")
                st.write(f"- Max głębokość [mm]: {np.nanmax(Z_mod):.1f}")
                st.write(f"- Liczba poprawnych punktów siatki: {np.count_nonzero(~np.isnan(Z_mod))}")

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania plików. Upewnij się, że oba pliki posiadają prawidłową strukturę. Szczegóły błędu: {e}")

else:
    # Komunikat startowy, gdy pliki nie zostały jeszcze wczytane
    st.info("👈 Aby rozpocząć analizę, prześlij oba pliki CSV (Oryginalny oraz Zmodyfikowany w skali mm) w panelu bocznym po lewej stronie.")