import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project AARAMBH | KAM Platform",
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
        is_critical = (i % 5 == 0) # Every 5th account is critical
        
        # Risk Logic
        churn_prob = random.randint(75, 95) if is_critical else random.randint(5, 45)
        risk_level = "Critical" if churn_prob > 70 else "Medium" if churn_prob > 40 else "Low"
        
        # Sentiment Logic
        sentiment = round(random.uniform(-0.9, -0.4), 2) if is_critical else round(random.uniform(0.1, 0.9), 2)
        
        # CLV Logic (Float for calculation)
        clv_raw = round(random.uniform(2, 18), 1)
        
        # Whitespace Logic
        active_products = random.sample(products, k=random.randint(2, 4))
        opportunity_product = random.choice([p for p in products if p not in active_products])
        
        # Topic Logic
        critical_topics = ["Pricing Structure", "SLA Breach", "Competitor Mention"]
        growth_topics = ["Expansion Plans", "GenAI Features", "API Integration"]
        primary_topic = random.choice(critical_topics) if is_critical else random.choice(growth_topics)
        
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
            "clv_val": clv_raw,
            "clv": f"${clv_raw}M",
            "admin_saved": f"{random.randint(2, 6)} hrs/wk",
            "active_products": active_products,
            "opportunity_product": opportunity_product,
            "opportunity_score": random.randint(75, 98),
            "primary_topic": primary_topic
        }
        data.append(account)
        
    return pd.DataFrame(data)

df = generate_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚡ Project AARAMBH")
    st.markdown("---")
    
    st.subheader("Navigation")
    # Add Portfolio Overview as the first option
    options = ["Portfolio Overview"] + df['name'].tolist()
    selected_option = st.selectbox("Select View", options)
    
    st.markdown("---")
    st.markdown("### 🤖 Agentforce Status")
    st.success("● AI Agents Active")
    st.caption("Monitoring 15 Accounts for Risk Signals")
    
    st.markdown("---")
    st.info(f"**User:** KAM_User\n\n**Role:** Strategic Account Manager")

# --- 4. MAIN DASHBOARD LOGIC ---

if selected_option == "Portfolio Overview":
    # ==========================================
    # PORTFOLIO OVERVIEW MODE
    # ==========================================
    
    st.title("📊 Portfolio Executive Summary")
    st.markdown("High-level strategic view of your book of business.")
    st.markdown("---")

    # A. KPI CARDS
    total_revenue = df['clv_val'].sum()
    avg_sentiment = df['sentiment_score'].mean()
    critical_count = df[df['risk_level'] == 'Critical'].shape[0]
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Total Portfolio Value (CLV)", f"${total_revenue:.1f}M", "+$1.2M (MoM)")
    with kpi2:
        st.metric("Avg. Customer Sentiment", f"{avg_sentiment:.2f}", "-0.05", delta_color="inverse")
    with kpi3:
        st.metric("Critical Risk Accounts", str(critical_count), f"{critical_count} Urgent", delta_color="inverse")
    with kpi4:
        st.metric("Avg. Churn Probability", f"{int(df['churn_prob'].mean())}%")

    st.markdown("---")

    # B. BROAD INSIGHTS & CALL TO ACTION
    c_insight, c_cta = st.columns([2, 1])
    
    with c_insight:
        st.subheader("💡 Broad Strategic Insights")
        st.info(f"**Retention Alert:** {critical_count} accounts are showing high churn probability (>70%). Primary driver identified as '{df[df['risk_level']=='Critical']['primary_topic'].mode()[0]}'.")
        st.success(f"**Growth Opportunity:** {len(df[df['nba_type']=='Growth'])} accounts are ripe for 'Agentforce' expansion based on whitespace analysis.")
        
    with c_cta:
        st.subheader("🚀 Primary Call to Action")
        top_critical = df.sort_values('churn_prob', ascending=False).iloc[0]
        st.markdown(f"""
        <div style="padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background-color: rgba(255, 75, 75, 0.1);">
            <strong>Urgent Intervention Required</strong><br>
            <u>Account:</u> {top_critical['name']}<br>
            <u>Action:</u> {top_critical['next_best_action']}<br>
            <u>Why:</u> Risk Score {top_critical['churn_prob']}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # C. WHO TO APPROACH FIRST (Risk Table) & KEY ISSUES
    row3_1, row3_2 = st.columns([2, 1])
    
    with row3_1:
        st.subheader("🔥 Priority List: Who to Approach First")
        
        # Styled Dataframe
        priority_df = df[['name', 'tier', 'risk_level', 'churn_prob', 'clv', 'next_best_action']].sort_values('churn_prob', ascending=False).head(5)
        
        # Color formatting function
        def highlight_risk(val):
            color = '#ff4b4b' if val > 70 else '#eab308' if val > 40 else '#22c55e'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            priority_df.style.map(highlight_risk, subset=['churn_prob']),
            use_container_width=True,
            column_config={
                "name": "Account Name",
                "risk_level": "Risk Tier",
                "churn_prob": st.column_config.ProgressColumn("Churn Risk (%)", format="%d%%", min_value=0, max_value=100),
                "next_best_action": "Recommended Action"
            },
            hide_index=True
        )

    with row3_2:
        st.subheader("🧩 Key Systemic Issues")
        # Topic Aggregation
        topic_counts = df['primary_topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        
        fig_pie = px.pie(topic_counts, values='Count', names='Topic', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    # ==========================================
    # INDIVIDUAL ACCOUNT MODE (Existing Logic)
    # ==========================================
    
    # Get selected account data
    account = df[df['name'] == selected_option].iloc[0]

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
            title = {'text': "Churn Probability"},
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
        st.markdown("#### Timeline Risk")
        st.metric("Time-to-Churn Est.", account['time_to_churn'])
        st.metric("Hazard Ratio", f"{account['hazard_ratio']}x", delta_color="inverse")
        
        st.markdown("**Top Risk Factors:**")
        if account['risk_level'] == "Critical":
            st.error("• Low License Utilization")
            st.error("• Sentiment Drop (Last 30d)")
        else:
            st.success("• High Feature Adoption")
            st.success("• Consistent Payment")

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
            "Topic": [account['primary_topic'], "Support Turnaround", "User Training"],
            "Volume": ["High", "Medium", "Low"],
            "Sentiment": ["Negative" if account['risk_level'] == "Critical" else "Positive", "Neutral", "Neutral"]
        })
        st.dataframe(topic_df, hide_index=True, use_container_width=True)

    with i_col2:
        st.markdown("#### Whitespace Map (KGAT)")
        st.info("Knowledge Graph Recommendation")
        
        st.write("**Current Active Products:**")
        st.write(", ".join(account['active_products']))
        
        st.markdown("---")
        
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
