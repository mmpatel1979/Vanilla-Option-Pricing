import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from streamlit import cache_data

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")

#Sidebar
st.sidebar.title("Option Parameters")
st.sidebar.markdown("---")
option_style = st.sidebar.selectbox("Style", ["European", "American"])
option_type = st.sidebar.radio("Type", ["Call", "Put"])

st.sidebar.markdown("---")
S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=110.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", value=0.5)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.06)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.3)
N = st.sidebar.slider("Steps (Tree Depth)", min_value=10, max_value=300, value=100)

st.sidebar.markdown("---")
spot_min = st.sidebar.number_input("Min Spot for Surface", value=S * 0.6)
spot_max = st.sidebar.number_input("Max Spot for Surface", value=S * 1.4)
vol_min = st.sidebar.number_input("Min Volatility", value=sigma * 0.5)
vol_max = st.sidebar.number_input("Max Volatility", value=sigma * 1.5)

#Pricing Model 
class BlackScholes:
    def __init__(self, T, S, K, r, sigma, option_type):
        self.T, self.S, self.K, self.r, self.sigma = T, S, K, r, sigma
        self.option_type = option_type

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'Call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

class Binomial:
    def __init__(self, T, S, K, r, sigma, N, option_type):
        self.T, self.S, self.K, self.r, self.sigma, self.N = T, S, K, r, sigma, int(N)
        self.option_type = option_type

    def price(self, style):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)

        ST = self.S * d ** np.arange(self.N, -1, -1) * u ** np.arange(0, self.N + 1)
        payoff = np.maximum(ST - self.K, 0) if self.option_type == 'Call' else np.maximum(self.K - ST, 0)

        for i in range(self.N):
            payoff = disc * (q * payoff[1:] + (1 - q) * payoff[:-1])
            if style == 'American':
                ST = self.S * d ** np.arange(self.N - i - 1, -1, -1) * u ** np.arange(0, self.N - i)
                exercise = np.maximum(ST - self.K, 0) if self.option_type == 'Call' else np.maximum(self.K - ST, 0)
                payoff = np.maximum(payoff, exercise)

        return payoff[0]

class Trinomial:
    def __init__(self, T, S0, K, r, sigma, N, option_type):
        self.T = T
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.N = int(N)
        self.option_type = option_type

    def price(self, style="European"):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        pu = ((np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd
        disc = np.exp(-self.r * dt)

        logu = np.log(u)
        center = self.N

        logS = np.array([np.log(self.S0) + (j - center) * logu for j in range(2 * self.N + 1)])
        S = np.exp(logS)

        # Payoff initialization depending on option type
        if self.option_type == "Call":
            V = np.maximum(S - self.K, 0)
        else:
            V = np.maximum(self.K - S, 0)

        for step in range(self.N, 0, -1):
            newV = np.zeros(2 * step - 1)
            for i in range(2 * step - 1):
                cont = disc * (pu * V[i] + pm * V[i + 1] + pd * V[i + 2])
                if style == "American":
                    newS = np.exp(np.log(self.S0) + (i - (step - 1)) * logu)
                    intrinsic = max(newS - self.K, 0) if self.option_type == "Call" else max(self.K - newS, 0)
                    newV[i] = max(cont, intrinsic)
                else:
                    newV[i] = cont
            V = newV

        return V[0]

#Greeks
class Greeks:
    def __init__(self, T, S, K, r, sigma, option_type):
        self.T, self.S, self.K, self.r, self.sigma = T, S, K, r, sigma
        self.option_type = option_type

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def delta(self):
        d1 = self.d1()
        return norm.cdf(d1) if self.option_type == 'Call' else norm.cdf(d1) - 1

    def gamma(self):
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def theta(self):
        d1, d2 = self.d1(), self.d2()
        term1 = -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) if self.option_type == 'Call' else \
                 self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return (term1 + term2) / 365

    def rho(self):
        d2 = self.d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2 if self.option_type == 'Call' else -d2) * 0.01

#Greeks Display
greeks = Greeks(T, S, K, r, sigma, option_type)
cols = st.columns(5)
cols[0].metric("Delta", f"{greeks.delta():.4f}")
cols[1].metric("Gamma", f"{greeks.gamma():.4f}")
cols[2].metric("Vega", f"{greeks.vega():.4f}")
cols[3].metric("Theta", f"{greeks.theta():.4f}")
cols[4].metric("Rho", f"{greeks.rho():.4f}")

@st.cache_data(show_spinner=True)
def compute_surface_grids(T, K, r, N, option_type, option_style, spot_min, spot_max, vol_min, vol_max):
    spot_vals = np.linspace(spot_min, spot_max, 40)
    vol_vals = np.linspace(vol_min, vol_max, 40)
    spot_grid, vol_grid = np.meshgrid(spot_vals, vol_vals)

    bs_grid = np.zeros_like(spot_grid)
    bin_grid = np.zeros_like(spot_grid)
    tri_grid = np.zeros_like(spot_grid)

    for i in range(spot_grid.shape[0]):
        for j in range(spot_grid.shape[1]):
            s_val = spot_grid[i, j]
            vol_val = vol_grid[i, j]

            bs_grid[i, j] = BlackScholes(T, s_val, K, r, vol_val, option_type).price()
            bin_grid[i, j] = Binomial(T, s_val, K, r, vol_val, N, option_type).price(option_style)
            tri_grid[i, j] = Trinomial(T, s_val, K, r, vol_val, N, option_type).price(option_style)

    return spot_vals, vol_vals, bs_grid, bin_grid, tri_grid

# Call the cached function
spot_vals, vol_vals, bs_grid, bin_grid, tri_grid = compute_surface_grids(
    T, K, r, N, option_type, option_style, spot_min, spot_max, vol_min, vol_max
)

#Plotly Heatmap 
def plotly_heatmap(z, x, y, title, current_spot, current_vol):
    # Determine z_value at current point for tooltip
    if "Black-Scholes" in title:
        z_value = BlackScholes(T, current_spot, K, r, current_vol, option_type).price()
    elif "Binomial" in title:
        z_value = Binomial(T, current_spot, K, r, current_vol, N, option_type).price(option_style)
    else:
        z_value = Trinomial(T, current_spot, K, r, current_vol, N, option_type).price(option_style)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar=dict(title='Option Price'),
    ))
    fig.add_trace(go.Scatter(
        x=[current_spot],
        y=[current_vol],
        mode='markers',
        marker=dict(color='white', size=12, symbol='circle', line=dict(color='black', width=1)),
        name='Current Params',
        hovertemplate='Option Price: %{text:.4f}<extra></extra>',
        text=[z_value]
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Spot Price',
        yaxis_title='Volatility',
        height=500,
        margin=dict(l=60, r=60, t=50, b=60),
    )
    return fig

#Show Plots
st.plotly_chart(plotly_heatmap(bs_grid, spot_vals, vol_vals, f"Black-Scholes {option_style} {option_type}", S, sigma), use_container_width=True)
st.plotly_chart(plotly_heatmap(bin_grid, spot_vals, vol_vals, f"Binomial {option_style} {option_type}", S, sigma), use_container_width=True)
st.plotly_chart(plotly_heatmap(tri_grid, spot_vals, vol_vals, f"Trinomial {option_style} {option_type}", S, sigma), use_container_width=True)

'streamlit run "/Users/mitpatel/Desktop/Projects/Vanilla Option Pricing/app.py"'
