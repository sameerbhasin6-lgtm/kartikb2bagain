import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project AARAMBH | KAM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA GENERATION ENGINE (Mock Data) ---
@st.cache_data
def generate_data():
    account_names = [
        "Alpha Solutions", "Beta Corp", "Gamma Systems", "Delta Dynamics", "Epsilon Energy", 
        "Zeta Finance", "Eta Healthcare", "Theta Retail", "Iota Logistics", "Kappa Tech", 
        "Lambda Auto", "Mu Media", "Nu Construction", "Xi Aerospace", "Omicron Cyber"
    ]
    industries = ['Financial Services', 'Retail', 'Manufacturing', 'Healthcare', 'Logistics', 'Energy', 'Tech']
    tiers = ['Enterprise', 'Strategic', 'Growth']
    products = ['Sales Cloud', 'Service Cloud', 'Marketing Cloud', 'Commerce Cloud', 'Tableau', 'MuleSoft', 'Data Cloud']

    data = []
    
    for i, name in enumerate(account_names):
        is_critical = (i % 5 == 0)
        
        # Risk Logic
        churn_prob = random.randint(75, 95) if is_critical else random.randint(5, 45)
        risk_level = "Critical" if churn_prob > 70 else "Medium" if churn_prob > 40 else "Low"
        
        # Sentiment Logic
        sentiment = round(random.uniform(-0.9, -0.4), 2) if is_critical else round(random.uniform(0.1, 0.9), 2)
        
        # Whitespace Logic
        active_products = random.sample(products, k=random.randint(2, 4))
        opportunity_product = random.choice([p for p in products if p not in active_products])
        
        account = {
            "id": f"acc_{i:03d}",
            "name": name,
            "industry": random.choice(industries),
            "tier": random.choice(tiers),
            "risk_level": risk_level,
            "churn_prob": churn_prob,
            "time_to_churn": f"{random.randint(30, 90)} Days" if is_critical else "> 365 Days",
            "hazard_ratio": round(churn_prob / 25, 1),
            "sentiment_score": sentiment,
            "sentiment_label": "Frustrated" if sentiment < -0.4 else "Advocate" if sentiment > 0.4 else "Neutral",
            "next_best_action": "Schedule Executive Review" if is_critical else "Pitch Agentforce Expansion",
            "nba_type": "Retention" if is_critical else "Growth",
            "clv": f"${round(random.uniform(2, 18), 1)}M",
            "admin_saved": f"{random.randint(2, 6)} hrs/wk",
            "active_products": active_products,
            "opportunity_product": opportunity_product,
            "opportunity_score": random.randint(75, 98)
        }
        data.append(account)
        
    return pd.DataFrame(data)

df = generate_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚡ Project AARAMBH")
    st.markdown("---")
    
    st.subheader("Select Account")
    # Create a formatted list for the dropdown
    account_options = df['name'].tolist()
    selected_account_name = st.selectbox("Search Account", account_options)
    
    # Get selected account data
    account = df[df['name'] == selected_account_name].iloc[0]
    
    st.markdown("### Portfolio Health")
    st.metric("Avg Portfolio Churn Risk", "12%", "-2.4%")
    st.progress(88)
    
    st.info(f"**User ID:** KAM_{random.randint(1000,9999)}\n\n**Role:** Strategic Account Manager")

# --- 4. MAIN DASHBOARD ---

# Header Section
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.title(f"{account['name']}")
    st.caption(f"{account['industry']} | {account['tier']} Tier | Renewal: Aug 2026")
with c2:
    st.metric("Projected CLV", account['clv'])
with c3:
    st.metric("Admin Time Saved", account['admin_saved'], delta="High Efficiency")

st.markdown("---")

# Row 1: Risk & Timeline
st.subheader("1. AI Risk Engine & Timeline Analysis")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Gauge Chart using Plotly
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = account['churn_prob'],
        title = {'text': "Churn Probability (XGBoost)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if account['churn_prob'] > 60 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}],
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("#### Timeline Risk (Cox Regression)")
    st.metric("Time-to-Churn Est.", account['time_to_churn'])
    st.metric("Hazard Ratio", f"{account['hazard_ratio']}x", delta_color="inverse")
    
    st.markdown("**Top SHAP Risk Factors:**")
    if account['risk_level'] == "Critical":
        st.error("• Low License Utilization (Sales Cloud)")
        st.error("• Sentiment Drop (Last 30d)")
    else:
        st.success("• High Feature Adoption")
        st.success("• Consistent Payment History")

with col3:
    st.markdown("#### Live Action Center (SalesRLAgent)")
    
    # Custom CSS card for NBA
    st.markdown(f"""
    <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid {'#ff4b4b' if account['nba_type'] == 'Retention' else '#00cc96'};">
        <h3 style="margin:0;">Recommended Action: {account['next_best_action']}</h3>
        <p style="color: gray; margin-bottom: 10px;">Type: <b>{account['nba_type']}</b> | Urgency: <b>{'High' if account['risk_level'] == 'Critical' else 'Normal'}</b></p>
        <p><i>Rationale: { 'High churn probability detected via XGBoost.' if account['risk_level'] == 'Critical' else 'High engagement with GenAI topics detected via BERTopic.' }</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("") # Spacer
    if st.button("🚀 Execute Action via Agentforce"):
        st.toast(f"Action '{account['next_best_action']}' initiated!", icon="✅")

st.markdown("---")

# Row 2: Intelligence & Growth
st.subheader("2. Intelligence & Whitespace Mapping")
i_col1, i_col2 = st.columns(2)

with i_col1:
    st.markdown("#### Real-Time Sentiment (BERT)")
    
    # Sentiment Metric
    sent_color = "red" if account['sentiment_score'] < 0 else "green"
    st.markdown(f"<h2 style='color:{sent_color}'>{account['sentiment_score']} ({account['sentiment_label']})</h2>", unsafe_allow_html=True)
    
    # Mock Trend Chart
    trend_data = pd.DataFrame({
        'Interaction': ['Email 1', 'Call 1', 'Chat', 'Email 2', 'Current'],
        'Sentiment': [
            account['sentiment_score'] + random.uniform(-0.1, 0.1) for _ in range(5)
        ]
    })
    fig_trend = px.line(trend_data, x='Interaction', y='Sentiment', markers=True, title="Last 5 Interactions Trend")
    fig_trend.update_layout(height=250)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### Top Topics (BERTopic)")
    topic_df = pd.DataFrame({
        "Topic": ["Pricing Structure", "GenAI Features", "Integration Support"],
        "Volume": ["High", "Medium", "Low"],
        "Sentiment": ["Negative" if account['risk_level'] == "Critical" else "Positive", "Neutral", "Neutral"]
    })
    st.dataframe(topic_df, hide_index=True, use_container_width=True)

with i_col2:
    st.markdown("#### Whitespace Map (KGAT)")
    st.info("Knowledge Graph Attention Network Recommendation")
    
    # Active Products
    st.write("**Current Active Products:**")
    st.write(", ".join(account['active_products']))
    
    st.markdown("---")
    
    # Opportunity Card
    st.success(f"**🎯 Top Opportunity: {account['opportunity_product']}**")
    
    c_metrics1, c_metrics2 = st.columns(2)
    with c_metrics1:
        st.metric("Propensity Score", f"{account['opportunity_score']}%")
    with c_metrics2:
        st.metric("Est. Revenue Impact", "$120k ARR")
        
    st.markdown(f"**Reasoning:** Peer adoption match. 85% of similar {account['industry']} clients utilize {account['opportunity_product']}.")

# --- 5. FOOTER ---
st.markdown("---")
st.markdown("© 2025 Project AARAMBH | Powered by Salesforce Data Cloud & Streamlit")