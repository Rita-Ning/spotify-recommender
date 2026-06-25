import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dotenv import load_dotenv
import os
load_dotenv()

# ========================================
# 設定 OpenAI API Key
# ========================================
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# ========================================
# 載入資料（只載入一次，用 cache 加速）
# ========================================
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    
    # 資料清洗
    df.dropna(subset=['artists', 'album_name', 'track_name'], inplace=True)
    df = df[df['duration_ms'] <= 600000]
    df = df[df['tempo'] != 0]
    df = df.reset_index(drop=True)
    
    # 特徵選取和標準化
    features = [
        'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, features

# ========================================
# 推薦函數
# ========================================
def get_recommendations(track_name, artist, df, X_scaled, n=10):
    matches = df[df['track_name'].str.lower() == track_name.lower()]
    
    if artist:
        matches = matches[matches['artists'].str.lower().str.contains(artist.lower())]
    
    if matches.empty:
        return None, None
    
    idx = matches.index[0]
    song_vector = X_scaled[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_vector, X_scaled)[0]
    sim_indices = sim_scores.argsort()[::-1]
    sim_indices = sim_indices[1:n+1]
    
    recommendations = df.loc[sim_indices, ['track_name', 'artists', 'track_genre']].copy()
    recommendations['similarity'] = sim_scores[sim_indices].round(3)

    # 去除重複歌曲！保留第一次出現的
    recommendations = recommendations.drop_duplicates(subset=['track_name', 'artists'])

    # 只保留需要的數量
    recommendations = recommendations.head(n)

    recommendations = recommendations.reset_index(drop=True)
    recommendations.index += 1
    
    return recommendations, df.loc[idx]

# ========================================
# AI 解釋函數
# ========================================
def explain_recommendation(input_song, input_artist, rec_song, rec_artist, df, features):
    input_features = df[
        (df['track_name'].str.lower() == input_song.lower()) &
        (df['artists'].str.lower().str.contains(input_artist.lower()))
    ][features].iloc[0]
    
    rec_features = df[
        df['track_name'].str.lower() == rec_song.lower()
    ][features].iloc[0]
    
    prompt = f"""
    你是一個音樂推薦專家。
    使用者喜歡：{input_song} by {input_artist}
    這首歌的特徵：
    - 舞蹈性: {input_features['danceability']:.2f}
    - 能量: {input_features['energy']:.2f}
    - 情緒正面程度: {input_features['valence']:.2f}
    - 速度: {input_features['tempo']:.1f} BPM
    
    推薦：{rec_song} by {rec_artist}
    這首歌的特徵：
    - 舞蹈性: {rec_features['danceability']:.2f}
    - 能量: {rec_features['energy']:.2f}
    - 情緒正面程度: {rec_features['valence']:.2f}
    - 速度: {rec_features['tempo']:.1f} BPM
    
    請用2-3句話解釋為什麼推薦這首歌，用輕鬆友善的語氣！
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    return response.choices[0].message.content

# ========================================
# Streamlit UI
# ========================================
st.title("🎵 Spotify 音樂推薦系統")
st.write("輸入你喜歡的歌曲，我們幫你找相似的音樂！")

# 載入資料
df, X_scaled, features = load_data()

# 輸入區域
col1, col2 = st.columns(2)
with col1:
    track_name = st.text_input("🎵 歌曲名稱", placeholder="例如：Shape of You")
with col2:
    artist_name = st.text_input("🎤 藝術家", placeholder="例如：Ed Sheeran")

n_recommendations = st.slider("推薦數量", min_value=3, max_value=10, value=5)

# 搜尋按鈕
if st.button("🔍 搜尋推薦"):
    if not track_name:
        st.warning("請輸入歌曲名稱！")
    else:
        with st.spinner("搜尋中..."):
            results, input_song = get_recommendations(
                track_name, artist_name, df, X_scaled, n_recommendations
            )
        
        if results is None:
            st.error(f"找不到歌曲：{track_name}，請確認名稱是否正確！")
        else:
            st.success(f"✅ 找到歌曲：{input_song['track_name']} - {input_song['artists']}")
            st.subheader("🎵 推薦歌曲：")
            st.dataframe(results)
            
            # AI 解釋 Top 3
            st.subheader("🤖 推薦原因：")
            top_n = min(3, len(results))
            for i in range(1, top_n + 1):
                rec_song = results.loc[i, 'track_name']
                rec_artist = results.loc[i, 'artists']
                
                with st.expander(f"#{i} {rec_song} - {rec_artist}"):
                    with st.spinner("AI 分析中..."):
                        explanation = explain_recommendation(
                            track_name, artist_name,
                            rec_song, rec_artist,
                            df, features
                        )
                    st.write(explanation)