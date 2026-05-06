import streamlit as st
import sympy as sp

# 1. Page Configuration
st.set_page_config(page_title="Laplace Solver", layout="centered")
st.title("⚡ Laplace Transform Solver")
st.write("A symbolic calculator for Differential Equations.")

# 2. Define Symbolic Variables
# We define t as positive and real so SymPy doesn't generate complex absolute values
t = sp.Symbol('t', real=True, positive=True)
s = sp.Symbol('s')

# Local dictionary to help SymPy parse text input easily
local_dict = {'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'u': sp.Heaviside}

# 3. Create the UI Tabs
tab1, tab2 = st.tabs(["Forward Transform f(t) ➔ F(s)", "Inverse Transform F(s) ➔ f(t)"])

# --- TAB 1: FORWARD TRANSFORM ---
with tab1:
    st.write("### Time Domain Input")
    # Default value is the e^-t * sin(t) example from your notes
    f_input = st.text_input("Enter a function f(t):", value="exp(-t) * sin(t)")
    
    try:
        # sympify converts the raw string into a mathematical object
        f_expr = sp.sympify(f_input, locals=local_dict)
        
        st.write("**Parsed Function $f(t)$:**")
        st.latex(sp.latex(f_expr))
        
        if st.button("Compute Laplace Transform F(s)"):
            with st.spinner("Transforming..."):
                # noconds=True removes the convergence conditions so the output is clean
                F_expr = sp.laplace_transform(f_expr, t, s, noconds=True)
                
                st.success("Transformation Complete!")
                st.write("**Laplace Transform $F(s)$:**")
                st.latex(sp.latex(F_expr))
                
    except Exception as e:
        st.error("Waiting for valid math input...")

# --- TAB 2: INVERSE TRANSFORM & PARTIAL FRACTIONS ---
with tab2:
    st.write("### Frequency Domain Input")
    # Default value is the heavy partial fraction example from your class notes
    F_input = st.text_input("Enter a function F(s):", value="(2*s + 1) / (s*(s+1)*(s^2 + 4*s + 6))")
    
    try:
        F_expr = sp.sympify(F_input, locals=local_dict)
        
        st.write("**Parsed Function $F(s)$:**")
        st.latex(sp.latex(F_expr))
        
        col1, col2 = st.columns(2)
        
        # Button 1: Partial Fractions
        with col1:
            if st.button("Decompose (Partial Fractions)"):
                with st.spinner("Factoring..."):
                    F_partial = sp.apart(F_expr, s)
                    st.write("**Decomposed:**")
                    st.latex(sp.latex(F_partial))
                    
        # Button 2: Inverse Transform
        with col2:
            if st.button("Compute Inverse f(t)"):
                with st.spinner("Inverting..."):
                    f_inv = sp.inverse_laplace_transform(F_expr, s, t)
                    st.success("Inversion Complete!")
                    st.write("**Inverse Transform $f(t)$:**")
                    st.latex(sp.latex(f_inv))
                    
    except Exception as e:
        st.error("Waiting for valid math input...")