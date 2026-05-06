import streamlit as st
import sympy as sp

# 1. Page Configuration
st.set_page_config(page_title="Laplace Solver", layout="centered")
st.title("⚡ Laplace Transform Solver")
st.write("A symbolic calculator for Differential Equations.")

# 2. Define Symbolic Variables
t = sp.Symbol('t', real=True, positive=True)
s = sp.Symbol('s')
local_dict = {'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'u': sp.Heaviside}

# 3. Create the UI Tabs
tab1, tab2, tab3 = st.tabs(["Forward Transform", "Inverse Transform", "Full ODE Solver"])

# --- TAB 1: FORWARD TRANSFORM ---
with tab1:
    st.write("### Time Domain Input")
    f_input = st.text_input("Enter a function f(t):", value="exp(-t) * sin(t)")
    
    try:
        f_expr = sp.sympify(f_input, locals=local_dict)
        st.write("**Parsed Function $f(t)$:**")
        st.latex(sp.latex(f_expr))
        
        if st.button("Compute Laplace Transform F(s)"):
            with st.spinner("Transforming..."):
                F_expr = sp.laplace_transform(f_expr, t, s, noconds=True)
                st.success("Transformation Complete!")
                st.write("**Laplace Transform $F(s)$:**")
                st.latex(sp.latex(F_expr))
    except Exception as e:
        st.error("Waiting for valid math input...")

# --- TAB 2: INVERSE TRANSFORM ---
with tab2:
    st.write("### Frequency Domain Input")
    F_input = st.text_input("Enter a function F(s):", value="1 / (s^2 + 4)")
    
    try:
        F_expr = sp.sympify(F_input, locals=local_dict)
        st.write("**Parsed Function $F(s)$:**")
        st.latex(sp.latex(F_expr))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Decompose (Partial Fractions)"):
                with st.spinner("Factoring..."):
                    F_partial = sp.apart(F_expr, s)
                    st.write("**Decomposed:**")
                    st.latex(sp.latex(F_partial))
        with col2:
            if st.button("Compute Inverse f(t)"):
                with st.spinner("Inverting..."):
                    f_inv = sp.inverse_laplace_transform(F_expr, s, t)
                    st.success("Inversion Complete!")
                    st.write("**Inverse Transform $f(t)$:**")
                    st.latex(sp.latex(f_inv))
    except Exception as e:
        st.error("Waiting for valid math input...")

# --- TAB 3: FULL ODE SOLVER (Your Step 4 Example) ---
with tab3:
    st.write("### Solve: $a y'' + b y' + c y = f(t)$")
    
    # Input columns for coefficients (Defaulted to y'' + 4y' + 6y)
    colA, colB, colC = st.columns(3)
    a = colA.number_input("a (y'' coeff)", value=1)
    b = colB.number_input("b (y' coeff)", value=4)
    c = colC.number_input("c (y coeff)", value=6)

    # Forcing function (Defaulted to 1 + e^-t)
    f_input_ode = st.text_input("Forcing Function f(t):", value="1 + exp(-t)", key="ode_f")

    # Initial Conditions (Defaulted to y(0)=0, y'(0)=0)
    colD, colE = st.columns(2)
    y0 = colD.number_input("y(0)", value=0)
    yprime0 = colE.number_input("y'(0)", value=0)

    if st.button("Solve Differential Equation"):
        with st.spinner("Calculating via Laplace..."):
            try:
                # 1. Transform the right side
                f_expr = sp.sympify(f_input_ode, locals=local_dict)
                F_s = sp.laplace_transform(f_expr, t, s, noconds=True)

                # 2. Build the algebraic left side
                char_poly = a*s**2 + b*s + c
                ic_terms = a*s*y0 + a*yprime0 + b*y0

                # 3. Solve for Y(s)
                Y_s = (F_s + ic_terms) / char_poly

                st.write("**Step 1: Transform to S-Domain $Y(s)$**")
                st.latex(sp.latex(Y_s))

                st.write("**Step 2: Partial Fraction Decomposition**")
                Y_partial = sp.apart(Y_s, s)
                st.latex(sp.latex(Y_partial))

                st.write("**Step 3: Inverse Transform to Time Domain $y(t)$**")
                y_t = sp.inverse_laplace_transform(Y_s, s, t)
                st.latex(sp.latex(y_t))
                
            except Exception as e:
                st.error("Error computing solution. Check your math syntax!")