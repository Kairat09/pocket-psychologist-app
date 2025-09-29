import streamlit as st
import torch
from transformers import pipeline
import pandas as pd
import random
import altair as alt
import os
from datetime import datetime

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
JOURNAL_FILE = "journal.csv"
EMOTION_COLORS = { "радость": "#4CAF50", "спокойствие": "#2196F3", "удивление": "#FFC107", "грусть": "#FFEB3B", "тревога": "#FF9800", "гнев": "#F44336" }
BASE_EMOTIONS = list(EMOTION_COLORS.keys())
# Вставьте сюда вашу полную базу RECOMMENDATIONS
RECOMMENDATIONS = {
    "радость": ["Отличный настрой! Попробуйте поделиться своей энергией с кем-то еще сегодня."],
    "спокойствие": ["Вы в состоянии гармонии. Это отличное время для медитации или спокойной прогулки."],
    "грусть": ["Кажется, вы немного грустите. Это нормально. Позвольте себе это почувствовать, не осуждая."],
    "тревога": ["Чувствуете беспокойство? Попробуйте сделать 5 глубоких вдохов и выдохов."],
    "гнев": ["Вы чувствуете гнев, и это сильная эмоция. Физическая активность может помочь."],
    "удивление": ["Что-то вас удивило! Мир полон неожиданностей."]
}

# --- ФУНКЦИИ ---
@st.cache_resource
def load_models():
    print("Загрузка моделей...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    emotion_classifier_zeroshot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    emotion_classifier_specialized = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, truncation=True)
    print("Все модели загружены.")
    return speech_recognizer, emotion_classifier_zeroshot, emotion_classifier_specialized

def get_recommendation(emotion_results):
    dominant_emotion = emotion_results['labels'][0]
    if dominant_emotion in RECOMMENDATIONS:
        return random.choice(RECOMMENDATIONS[dominant_emotion])
    return "Постарайтесь прислушаться к себе."

def save_to_journal(text, zeroshot_results, specialized_emotions, recommendation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_scores = {label: 0.0 for label in BASE_EMOTIONS}
    for label, score in zip(zeroshot_results['labels'], zeroshot_results['scores']):
        if label in emotion_scores:
            emotion_scores[label] = score
    df_zeroshot = pd.DataFrame({'label': zeroshot_results['labels'], 'score': zeroshot_results['scores']})
    top_3_zeroshot = ", ".join([f"{row['label']} ({row['score']:.2f})" for index, row in df_zeroshot.head(3).iterrows()])
    top_3_goemotions = ", ".join([f"{e['label']} ({e['score']:.2f})" for e in specialized_emotions[0][:3]])
    new_entry = {
        "Дата и время": timestamp, "Распознанный текст": text, "Выданный совет": recommendation,
        "ТОП-3 (6 базовых эмоций)": top_3_zeroshot, "ТОП-3 (GoEmotions)": top_3_goemotions, **emotion_scores
    }
    df_new = pd.DataFrame([new_entry])
    if not os.path.exists(JOURNAL_FILE):
        df_new.to_csv(JOURNAL_FILE, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(JOURNAL_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# --- ГЛАВНОЕ ПРИЛОЖЕНИЕ ---
speech_recognizer, emotion_classifier_zeroshot, emotion_classifier_specialized = load_models()

st.set_page_config(layout="wide")
st.title("🎧 Карманный психолог")

# --- CSS-ТРЮК ДЛЯ ВЫСОТЫ СТРОК ---
st.markdown("""
<style>
    .stDataFrame div[data-testid="stVerticalBlock"] {
        white-space: normal !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

tab1, tab2 = st.tabs(["Анализ сессии", "Мой журнал"])

# --- ВКЛАДКА 1: АНАЛИЗ СЕССИИ ---
with tab1:
    st.header("Проанализировать новое голосовое сообщение")
    uploaded_file = st.file_uploader("Выберите аудиофайл (.wav, .mp3)", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Анализировать"):
            with st.spinner("Анализ... Это может занять некоторое время..."):
                audio_bytes = uploaded_file.getvalue()
                transcribed_text = speech_recognizer(audio_bytes, return_timestamps=True)["text"]
                emotion_results_zeroshot = emotion_classifier_zeroshot(transcribed_text, BASE_EMOTIONS)
                emotion_results_specialized = emotion_classifier_specialized(transcribed_text)
                st.session_state.analysis_results = {
                    "text": transcribed_text, "zeroshot": emotion_results_zeroshot,
                    "specialized": emotion_results_specialized
                }

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.header("Результаты Анализа")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Анализ (Универсальная модель)")
            chart_data1 = pd.DataFrame({'Эмоция': results['zeroshot']['labels'], 'Вероятность': results['zeroshot']['scores']})
            chart1 = alt.Chart(chart_data1).mark_bar().encode(x='Вероятность', y=alt.Y('Эмоция', sort='-x'), color=alt.Color('Эмоция', scale=alt.Scale(domain=list(EMOTION_COLORS.keys()), range=list(EMOTION_COLORS.values())), legend=None)).properties(title='6 базовых эмоций')
            st.altair_chart(chart1, use_container_width=True)
        with col2:
            st.subheader("Анализ (Специализированная модель GoEmotions)")
            chart_data2 = pd.DataFrame(results['specialized'][0]).rename(columns={'label': 'Эмоция', 'score': 'Вероятность'}).head(7)
            chart2 = alt.Chart(chart_data2).mark_bar().encode(x='Вероятность', y=alt.Y('Эмоция', sort='-x')).properties(title='ТОП-7 из 28 эмоций')
            st.altair_chart(chart2, use_container_width=True)
        st.info(f"**Распознанный текст:** {results['text']}")
        st.divider()
        recommendation = get_recommendation(results['zeroshot'])
        st.subheader("💡 Мой вам совет на сегодня:")
        st.success(recommendation)
        if st.button("Сохранить в журнал"):
            save_to_journal(results['text'], results['zeroshot'], results['specialized'], recommendation)
            st.success("Запись успешно сохранена в вашем журнале!")
            st.session_state.analysis_results = None 
            st.rerun()

# --- ВКЛАДКА 2: МОЙ ЖУРНАЛ (с финальной, надежной версией таблицы) ---
with tab2:
    st.header("История ваших записей и динамика настроения")
    
    if os.path.exists(JOURNAL_FILE):
        journal_df = pd.read_csv(JOURNAL_FILE)
        
        st.subheader("Все записи")
        
        # --- ВОЗВРАЩАЕМСЯ К st.dataframe С НОВОЙ КОНФИГУРАЦИЕЙ ---
        display_df = journal_df[[
            "Дата и время", 
            "Распознанный текст", 
            "Выданный совет", 
            "ТОП-3 (6 базовых эмоций)", 
            "ТОП-3 (GoEmotions)"
        ]]

        st.dataframe(
            display_df,
            # Конфигурируем ширину колонок, чтобы дать больше места тексту
            column_config={
                "Распознанный текст": st.column_config.TextColumn(width="large"),
                "Выданный совет": st.column_config.TextColumn(width="medium"),
            },
            use_container_width=True, 
            hide_index=True
        )
        
        st.divider()

        if len(journal_df) > 1:
            st.subheader("Динамика вашего настроения")
            # ... (код для графика динамики остается без изменений) ...
            journal_df['Дата и время'] = pd.to_datetime(journal_df['Дата и время'])
            emotions_over_time = journal_df[['Дата и время'] + BASE_EMOTIONS]
            df_melted = emotions_over_time.melt('Дата и время', var_name='Эмоция', value_name='Вероятность')
            trend_chart = alt.Chart(df_melted).mark_line(point=True).encode(
                x=alt.X('Дата и время:T', title='Дата'),
                y=alt.Y('Вероятность:Q', title='Сила эмоции'),
                color=alt.Color('Эмоция:N', title='Эмоция', scale=alt.Scale(domain=list(EMOTION_COLORS.keys()), range=list(EMOTION_COLORS.values()))),
                tooltip=['Дата и время', 'Эмоция', 'Вероятность']
            ).properties(title='Изменение эмоций по дням').interactive()
            st.altair_chart(trend_chart, use_container_width=True)
        else:
            st.info("Сохраните хотя бы две записи, чтобы увидеть график динамики.")
    else:
        st.info("Ваш журнал пока пуст.")