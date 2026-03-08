gender = st.selectbox("Gender", ["Male","Female"])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])

tenure = st.number_input("Tenure")

phoneservice = st.selectbox("Phone Service", ["Yes","No"])
internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

monthlycharges = st.number_input("Monthly Charges")
totalcharges = st.number_input("Total Charges")