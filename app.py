import wolframalpha
import streamlit as st

def send(query):
    app_id = "J78J28-57PL462PJU"

    client = wolframalpha.Client(app_id)

    response = client.query(query)

    if hasattr(response, 'pods'):
        for pod in response.pods:
            if pod.title in ["Alternate form","Alternate forms","Roots","Root", "Solution", "Solutions","Properties as a real function", "Derivative", "Indefinate integral", "Identities", "Global minima", "Global maxima", "Alternative representations", "Integral representations", "Alternative representation", "Integral representation", "Definite integral over a half-period", "Limit", "Limits","Definite integral", "Indefinite integral","Local minimum", "Local minimums", "Local maximum", "Local maximums", "Polynomial discriminant", "Result", "Sum convergence", "Partial sum formula", "Regularized result", "Decimal approximation", "Constant name", "Prime factorization", "Divisors", "Property", "Real solutions", "Complex solutions", "Integer solution", "Exact result", "Alternate complex forms", "Polar coordinates", "Numerical root"] or pod.title.startswith("All") or pod.title.startswith("Series") or pod.title.startswith("Input"):
                st.text(pod.title)
                for subpod in pod.subpods:
                    if hasattr(subpod, 'plaintext'):
                        if subpod.title != "":
                            st.text(subpod.title + ":")
                        expr = (((((((((((((subpod.plaintext.replace("element","∈")).replace("for", "∀")).replace("sum", "\sum")).replace("integral", "\int")).replace("->", "→")).replace("i x", "{ix}")).replace("!=", "≠)")).replace(">=", "≥")).replace("<=", "≤")).replace("as\suming", "assuming")).replace("polygamma", "ψ")).replace("gamma", "Γ")).replace("product", "\prod")).replace("lim", "\lim")
                        while expr.find("_(") != -1:
                            expr = expr[:expr.find("_(")]+"_{"+expr[expr.find("_(")+2:].replace(")", "}", 1)
                        while expr.find("^(") != -1:
                            expr = expr[:expr.find("^(")]+"^{"+expr[expr.find("^(")+2:].replace(")", "}", 1)
                        expr = expr.replace(" ", "\hspace{1mm}")
                        for i in range(expr.count("sqrt(")):
                            p = expr.find("sqrt(")
                            expr = expr[:p] + "\sqrt{" + expr[p+5:]
                            cnt = 1
                            ite = p+5
                            while cnt > 0:
                                if expr[ite] == "(":
                                    cnt += 1
                                    ite += 1
                                elif expr[ite] == ")":
                                    cnt -= 1
                                    if cnt == 0:
                                        expr = expr[:ite]+"}"+expr[ite+1:]
                                        break
                                    ite += 1
                                else:
                                    ite += 1
                        for i in range(expr.count("abs(")):
                            p = expr.find("abs(")
                            expr = expr[:p] + "|" + expr[p+4:]
                            cnt = 1
                            ite = p+4
                            while cnt > 0:
                                if expr[ite] == "(":
                                    cnt += 1
                                    ite += 1
                                elif expr[ite] == ")":
                                    cnt -= 1
                                    if cnt == 0:
                                        expr = expr[:ite]+"|"+expr[ite+1:]
                                        break
                                    ite += 1
                                else:
                                    ite += 1
                        for i in range(expr.count("cuberoot(")):
                            p = expr.find("cuberoot(")
                            expr = expr[:p] + "\sqrt[3]{" + expr[p+9:]
                            cnt = 1
                            ite = p+9
                            while cnt > 0:
                                if expr[ite] == "(":
                                    cnt += 1
                                    ite += 1
                                elif expr[ite] == ")":
                                    cnt -= 1
                                    if cnt == 0:
                                        expr = expr[:ite]+"}"+expr[ite+1:]
                                        break
                                    ite += 1
                                else:
                                    ite += 1
                        for i in range(expr.count("floor(")):
                            p = expr.find("floor(")
                            expr = expr[:p] + r"\lfloor{" + expr[p+6:]
                            cnt = 1
                            ite = p+6
                            while cnt > 0:
                                if expr[ite] == "(":
                                    cnt += 1
                                    ite += 1
                                elif expr[ite] == ")":
                                    cnt -= 1
                                    if cnt == 0:
                                        expr = expr[:ite]+r"}\rfloor"+expr[ite+1:]
                                        break
                                    ite += 1
                                else:
                                    ite += 1
                        for i in range(expr.count("ceiling(")):
                            p = expr.find("ceiling(")
                            expr = expr[:p] + r"\lceil{" + expr[p+8:]
                            cnt = 1
                            ite = p+8
                            while cnt > 0:
                                if expr[ite] == "(":
                                    cnt += 1
                                    ite += 1
                                elif expr[ite] == ")":
                                    cnt -= 1
                                    if cnt == 0:
                                        expr = expr[:ite]+r"}\rceil"+expr[ite+1:]
                                        break
                                    ite += 1
                                else:
                                    ite += 1
                        for i in range(expr.count('log')):
                            p = expr.find("log(")


                        st.latex(expr)
                        
                st.text("-"*100)
    else:
        st.text("No results found")

st.set_page_config(
    page_title="Aziz Bot",
    page_icon="img/favicon.png"
)

def main():
    st.title("Math AI:")
    expression = st.text_input("Enter the expression: (Required)")
    if st.button("Sumbit"):
        send(expression)
        

if __name__ == "__main__":
    main()