# -*- coding: utf-8 -*-
"""
Spyder Editor

Requirements:
   environment named 'streamlit_env' created with Anaconda prompt:(base) conda create --name streamlit_env
   environment just created actived:(base) conda activate streamlit_env
   Then Streamlit installed:(streamlit_env) conda install -c conda-forge streamlit
   Installed version checked:(streamlit_env) streamlit --version
   Matplotlib installed:(streamlit_env) conda install -c conda-forge matplotlib
   Numpy installed:(streamlit_env) conda install -c conda-forge numpy
   
@author: S. Bailleux
"""
#For general help on Markdown text formatting suh as in st.markdown, consult the web page:
# https://www.eddymens.com/tags/markdown-tutorial
# In particular: """a line terminating with 2 subsequent spaces like this line is understood as a newline,  
#                providing that the string is written on 2 or more lines that are wrapped using 3": """

# Import necessary libraries
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from matplotlib.patches import Polygon
# For text annotations:
#from matplotlib.patches import Ellipse
#from matplotlib.text import OffsetFrom
# Enable Matplotlib to use LaTeX for text rendering
# See for examples: https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
#
from matplotlib import rcParams
#rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'


st.set_page_config(
    page_title="Thermodynamics app-ULille",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        #'Get Help': "https://sciences-technologies.univ-lille.fr/physique",
        'Report a bug': "mailto: stephane.bailleux@univ-lille.fr",
        'About': """An interactive app to better understand the fundamental concepts in *thermodynamics*.  
                    Designed for undergraduates in *physics* and their teachers.  
                    ©2023 T. Benabbas, A. Anakkar, **S. Bailleux**"""
    }
)


# Create a Streamlit app
# st.title("Triangular reversible thermodynamic cycle with infinithermal process")

# Some global variables
NPts = 16385        # Number of data points for each curve, computed as NPts =2^k +1, and hence:2NPts -2 =2^q +1 with q =k +1
                    # Such a value gives accurate area values using Simpson integration method, to within a few thousandths, which is quite enough.
Rmol = 8.314462618  # NIST value
gmIG = 7/5          # gamma for an Ideal Gas (5/3 or 7/5)
# global variables not defined yet (hence using numpy to set them as 'NaN'):
initV0 = 100.0
initP0 = 25000.0
initT0 = initT0 = initP0 * initV0/1000 /Rmol  # Vol. converted to m³
qValue = 2.50
kValue = 4.00
#tempMax = np.nan
tempMax = (initP0 *initV0/1000 /Rmol) /4 * (qValue *kValue -1)**2 /(kValue -1) /(qValue -1)
volTMax = initV0 / 2 * (qValue * kValue - 1) / (kValue - 1)
volSMax = 2 *volTMax * gmIG/(gmIG +1)
Eta=np.nan


# Create a layout with two columns: can be used to control the placement and size
# of widgets by adjusting their position in columns or containers.
# I can use the st.sidebar for a separate sidebar or create custom layouts using
# Streamlit's layout primitives like st.column, st.container, etc.
# col1, col2 = st.columns(2)
# Create a layout with two columns
#dataL_column, plotR_column = st.columns([.5, 5.5])
# Formtat the title columns with custom font size using Markdown
# dataL_column.markdown("# <span style='font-size:25px;'>State var.</span>", unsafe_allow_html=True)
#plotR_column.markdown(
#    "# <span style='font-size:25px;'>Triangular reversible engine cycle</span>", unsafe_allow_html=True)

# Left column: Display numerical data
#dataL_column.title('State var.')
# data = np.random.rand(100, 3)  # Sample numerical data
# dataL_column.write(data)

# Right column: Display plots in a scrollable area

# Option: set min. height of the right column for scrollability
# This ensures plots are displayed within a scrollable area if specified height is exceeded
# plotR_column.set_min_height(400)

# Create Tabs for the left Column
# tab1, tab2, tab3, tab4 = st.tabs(["State var.", "Data", "Efficiency', 'Help'])

# Placeholder variables for tab content
tab4_helpContent = r"""
**This interactive app** allows the student (and teacher) to study 1 mole of an *ideal* gas which follows
a reversible cyclic triangular process (in the clockwise or engine direction).
This triangular cycle (in the *p*, *V* diagram) starts with an isochoric change whose final pressure is
determined by *k*, and ends with an isobaric change whose initial volume is set by *q*.
Both *q* and *k* are unitless parameters verifying: 0≤{*q*, *k*}.
In between, the gas undergoes an *'infiniterm'* process consisting in a linear decrease of *p*(*V*):
*p=−aV+b*, where *a*=$\sf{\frac{k-1}{q-1}} \ \sf{\frac{p_0}{V_0}}$ Pa/m$^3$ and *b*=*p*$_0 \sf{\frac{qk - 1}{q-1}}$ Pa.
The initial state is A(*V*$_0$, *p*$_0$, *T*$_0$).
Switching to the 'State' tab, in addition to *γ* (=*C*$\mathrm{_p}$/*C*$\mathrm{_V}$, the specific heat ratio of a gas),
you can adjust the thermodyn. state variables for A, B and C.
Naturally the temperature of the gas changes along each path; but during the
*infinitherm* process and for certain values of *(q*, *k*), the temperature reaches a maximum
at one point, before decreasing. This is well visualized when scrutinizing the *T*, *V* plot.
Now observe the *Temperature vs Entropy* diagram, and set (*q*, *k*)=(2.50, 4.00) for *γ*=7/5 or (*q*, *k*)=(2.00, 3.50)
for *γ*=5/3: at point *N* the funny thing is that the system has reached its maximum of *entropy* whilst the temperature
already started decreasing. But that's not the end of the story.
Indeed, for such non-unique peculiar (*q*, *k*) couple of values, the global thermal transfer exchanged
between B and C is nul (look at *Q*$\sf{_{BC}}$ in 'Tables' tab and come back here) whilst the gas did gain some thermal
energy before releasing the whole of it: this infinitherm process is **NOT** adiabatic!

**In short, this app helps** distinguish the notion of temperature *T* (in K) from those of entropy *S* (in J/K)
and of thermal transfer *Q* (in J). In addition, to determine the efficiency of a cyclic process, it is crucial to
consider *local* changes (using integral calculus), since computations based on *global* exchanges yield erroneous results!
"""
tab4_Aide = r"""
**Cette app** ***interactive*** permet (à l'étudiant comme à l'enseignant) l'étude d'une (1) mole
d'un gaz *idéal* subissant un cycle réversible triangulaire moteur.
Ce cycle triangulaire (dans le diagramme *p*, *V*) débute par une transformation isochorique dont la pression finale est
déterminée par *k*, et se termine par une isobare dont le volume initial est fixé par *q*.
Ces deux paramètres adimensionnels vérifient : 0 ≤ {*q*, *k*}.
Entre ces deux transformations, le gaz subit un processus *'infinitherme'* consistant en une diminution linéaire de *p*(*V*) :
*p=−aV+b* avec *a* = $\sf{\frac{k-1}{q-1}} \ \sf{\frac{p_0}{V_0}}$ Pa/m$^3$ et *b* = *p*$_0 \sf{\frac{qk - 1}{q-1}}$ Pa.
L'état initial est : A(*V*$_0$, *p*$_0$, *T*$_0$).
En passant à l'onglet 'State', en plus de *γ* (=*C*$\mathrm{_p}$/*C*$\mathrm{_V}$, rapport des capacités thermiques à pression
constante et à volume constant du gaz), les variables d'état thermodynamique de A, B et C peuvent être ajustées.
Naturellement, la température du gaz varie le long de chaque transformation. Mais pendant le processus *infinitherme* et
pour certaines valeurs du couple (*q*, *k*) la température atteint un maximum en un point avant de décroître !
Ce phénomène est bien visible en examinant le diagramme *T*, *V*.
Observez maintenant le diagramme *T*, *S*, et fixez (*q*, *k*) = (2.50, 4.00) si *γ*=7/5 ou (*q*, *k*)=(2.00, 3.50) si *γ*=5/3:
au point *N*, il est original de constater que le système a atteint son maximum d'*entropie* alors que la température a déjà
commencé à diminuer ! Mais ce n'est pas la fin de l'histoire.
En effet, pour ce couple de valeurs (*q*, *k*) non unique, le transfert thermique *global* échangé entre B et C est nul (cf.
*Q*$\sf{_{BC}}$ dans l'onglet 'Tables') alors que le gaz a bien reçu de l'énergie thermique avant d'en céder la totalité :
ce processus *infinitherme* n'est **PAS** adiabatique !

**En résumé, cette** ***app*** permet de distinguer la notion de température *T* (en K) de celles d'entropie *S* (en J/K) et de
transfert thermique *Q* (en J).
De plus, pour déterminer l'efficacité d'un processus cyclique, il est crucial de considérer les changements *locaux*
(en utilisant le calcul intégral), car ceux basés sur des échanges *globaux* donnent des résultats erronés !
"""


# Compute temperature functions for each process of the triangular cycle, assuming 1 mol of an ideal gas as the system
st.cache_data(ttl=1200)
def defTemp(volume, pressure):
    return volume/1000 * pressure / Rmol

st.cache_data(ttl=1200)
def defMaxData(initP0, initV0, qValue, kValue, gmIG):
    # Considering the infinitherm process
    # Compute the maximum temperature reached
    #global tempMax
    tempMax = (initP0 *initV0/1000 /Rmol) /4 *(qValue *kValue -1)**2 /(kValue -1) /(qValue -1)

    # Compute gas volume at T = tempMAx
    #global volTMax
    volTMax = initV0 / 2 * (qValue * kValue - 1) / (kValue - 1)

    # Compute gas volume when entropie S is maximal
    #global volSMax
    volSMax = 2 * volTMax * gmIG/(gmIG + 1)

    initT0 = initP0 * initV0/1000 / Rmol

    # The point N has (Smax,volSmax) as coordinates. The Temperature of this point is tempNMax
    # print(volTMax, tempMax)
    # Calcuate some statistics about infinitherm process
    eNtrpyAB, eNtrpyBC, eNtrpyCA = eNtrpyEvol(initV0, initP0, initT0, qValue, kValue, gmIG)
    SMax = np.max(eNtrpyBC)
    # Get the index at which eNtropyBC is maximum, then get the temperature at this maximum of Entropy
#    index1 = np.abs(eNtrpyBC - SMax).argmin()
    volIsobaric, volIsoKoric, volInfnThrm = VolumeEvol(initV0, qValue)
    pIsobaric, pIsoKoric, pInfnThrm = PressureEvol(initP0, kValue)
    temInfnThrm = defTemp(volInfnThrm, pInfnThrm)
    global TNMax
    global STMax
    # Get the entropy value at maximum temperature TMax, STMax
    index2 = np.abs(temInfnThrm - tempMax).argmin()
    STMax = eNtrpyBC[index2]

#    if (initV0 <= volSMax <= qValue *initV0):
#        TNMax = temInfnThrm[index1]
        # Get the entropy value at maximum temperature TMax, STMax
        # TMax is here defined to be a global variable:
     #    global TMax
     #    TMax = initT0/4 * (qValue * kValue - 1)**2 / (qValue-1) / (kValue - 1)

#    else:
        #Determine TNMax: the index for which eXtendedVol=volSMax has to be determined
        #                 Knowing that eXtendedVol varies linearly, i.e.:
        #                 eXtendedVol=indeX/(2 NPts -2) *2 *volTMax with index varying from 0 to (2 NPts -2),
        #                 and knowing that volSMax=2 *volTMax *gmIG /(gmIG +1), then the index we want to compute is:
        #                 newIndx=gmIG /(gmIG +1) *(2 *NPts -2)). This is a floating-point number, which has to be converted
        #                 to an integer AFTER rounding it to the nearest value (int() function ony truncates the decimal part).
    eXtendedVol = np.linspace(0, 2 *volTMax, 2 *NPts-1)
    eXtendedPre = np.linspace(initP0 * (kValue *qValue -1) / (qValue -1), 0, 2*NPts-1)
    eXtendedTem = eXtendedVol/1000 * eXtendedPre / initT0 / Rmol
    newIndx = int( np.round(gmIG /(gmIG +1) *(2 *NPts -2)))
    TNMax = eXtendedTem[newIndx] *initT0

    return tempMax, volTMax, volSMax, SMax, TNMax, STMax

st.cache_data(ttl=1200)
def defXtendedData(initP0, initT0, qValue, kValue, volTMax):
    # Considering the infinitherm process:
    # Compute the eXtended range of volume: 0-2volTMax
    eXtendedVol = np.linspace(0, 2 *volTMax, 2 *NPts-1)

    # Compute the eXtended range of pressure
    eXtendedPre = np.linspace(initP0 * (kValue *qValue -1) / (qValue -1), 0, 2*NPts-1)

    # Compute the eXtended range of temperature: T(nul volume)-T(2volTMax)
    eXtendedTem = eXtendedVol/1000 * eXtendedPre / initT0 / Rmol

    # Return the above three 'waves'
    return eXtendedVol, eXtendedPre, eXtendedTem

st.cache_data(ttl=1200)
def PressureEvol(initP0, kValue):
    # Compute the Pressure evolution along the three processes
    pIsobaric = np.linspace(initP0, initP0, NPts)
    pIsoKoric = np.linspace(initP0, kValue * initP0, NPts)
    pInfnThrm = np.linspace(kValue * initP0, initP0, NPts)
    return pIsobaric, pIsoKoric, pInfnThrm

st.cache_data(ttl=1200)
def VolumeEvol(initV0, qValue):
    # Compute the Volume evolution along the three processes
    volIsobaric = np.linspace(initV0, qValue * initV0, NPts)
    volIsoKoric = np.linspace(initV0, initV0, NPts)
    volInfnThrm = np.linspace(initV0, qValue * initV0, NPts)
    return volIsobaric, volIsoKoric, volInfnThrm

st.cache_data(ttl=1200)
def eNtrpyEvol(initV0, initP0, initT0, qValue, kValue, gmIG):
    # Function to compute the Entropy evolution along the three processes

    volIsobaric, volIsoKoric, volInfnThrm = VolumeEvol(initV0, qValue)
    pIsobaric, pIsoKoric, pInfnThrm = PressureEvol(initP0, kValue)

    # Entropy Sab using S(A)=0 as Origin of Entropy
    # dSab = Cv dT/T; ΔSab = Cv Ln(T/To) = +Rmol 1/(γ-1) Ln(T/To)
    temIsoKoric = defTemp(volIsoKoric, pIsoKoric)
    eNtrpyAB = Rmol /(gmIG -1) * np.log(temIsoKoric /initT0)

    # Entropy Sbc to be computed during the infinitherm process
    # There is NO analytic expression (at least simple one) for T(S) for the infinitherm process. (Whilst T(S) is an
    # exponantiel growth both for the isobaric & isochoric process), which  means S(T) is proportional to Ln(T/To).)
    # Hence, since during the BC process: P=-aV+b, first the Volume is computed, then the Pressure is computed using: P=-aV +b,
    # afterwards the temperature is deduced using: PV = R T, and finally the Entropy is inferred!
    # The Volume varies linearly with point Number from V(B)=Vo to V(C)=qVo, it is given by: volInfnThrm
    # Pressure: varies linearly with point Number from P(B)=kPo to P(C)=Po, it is given by: pInfnThrm
    # Temperature, for 1 mol of an ideal Gas, reduced to the ration T/To, it is given by:
    temInfnThrm = defTemp(volInfnThrm, pInfnThrm)
    # Entropy: dSbc = Cv dT/T +R dV/V; We integrate for T varying from T(B)=kTo to T,and for V varying from V(B)=Vo to V
    # Therefore: ΔSbc = Cv [Ln(T)] + R [Ln(V)] (upper limits: state variables (T, V)  / lower limits (point B): kTo, Vo
    # Therefore: ΔSbc = Cv Ln(T/kTo) + R Ln(V/Vo) using: Cv=R/(γ-1).
    # Finally: Since the above computed value (ΔSbc) is the change in Entropy starting from point B, it is needed to shift
    #          this value (i.e. to change its origin) by adding the entropy for B computed during the AB process.
    # 		   This shift in origin is: eNtrpyAB[NPts -1]
    eNtrpyBC = eNtrpyAB[NPts - 1] + Rmol * \
        (np.log(temInfnThrm /initT0 /kValue) /
         (gmIG -1) + np.log(volInfnThrm/initV0))
    #print(eNtrpyAB[NPts -1])
    # Entropy Sca to be computed during the isobaric process
    # Be careful with shift due to entropy origin! Rmol *gmIG/(gmIG -1) *(0 +Ln(TempEvol_CA))
    # dSca = Cp dT/T; ΔSca = Cp Ln(To/T) = -Rmol γ/(γ-1) Ln(T/To)
    temIsobaric = defTemp(volIsobaric, pIsobaric)
    eNtrpyCA = Rmol *gmIG /(gmIG -1) * (0 + np.log(temIsobaric/initT0))

    return eNtrpyAB, eNtrpyBC, eNtrpyCA


st.cache_data(ttl=1200)
def wQuS(initV0, initP0, qValue, kValue, gmIG):
    # Compute work W, heat Q, intEgy DU, eNtrpy DS
    global ABWork, BCWork, CAWork
    global ABHeat, BCHeat, CAHeat
    global ABintEgy, BCintEgy, CAintEgy
    global ABeNtrpy, BCeNtrpy, CAeNtrpy

    initT0 = initP0 * initV0/1000 / Rmol

    #Work computations
    ABWork = 0 /1000
    BCWork = -Rmol *initT0 /2 *(qValue -1) *(kValue +1) /1000
    CAWork = +Rmol *initT0 *(qValue -1) /1000

	#Heat computations
    ABHeat =  Rmol *initT0 /(gmIG -1) *(kValue -1) /1000
    BCHeat =  Rmol *initT0 *( (qValue -kValue) /(gmIG -1) + (qValue -1)*(kValue +1)/2) /1000
    CAHeat = -Rmol *initT0 *gmIG/(gmIG -1)*(qValue -1) /1000

	#int. energy computations
    ABintEgy =  ABHeat
    BCintEgy =  Rmol *initT0 *(qValue -kValue) /(gmIG -1) /1000
    CAintEgy = -Rmol *initT0 /(gmIG -1) *(qValue -1) /1000

	#Entroy computations
    ABeNtrpy =  Rmol/(gmIG -1) *np.log(kValue)
    BCeNtrpy =  Rmol/(gmIG -1) *(gmIG *np.log(qValue) -np.log(kValue))
    CAeNtrpy = -gmIG/(gmIG -1) *Rmol *np.log(qValue)

st.cache_data(ttl=1200)
def areaWH(press1, press2, press3, vol1, vol2, vol3, temp1, eNtrpy1, temp2, eNtrpy2, temp3, eNtrpy3):
    areaW1 = np.trapz(press1, vol1)
    areaW2 = np.trapz(press2, vol2)
    areaW3 = np.trapz(press3, vol3)
    
    #Below: needs to compute only the postive area below the T vs S curve!
    #Along AB/Isochoric: the area is *always* positive (q>1 hence x_B > x_A) so returning areaH1 *IS* necessary!
    #Along CA/Isobaric:  the area is *always* negative (q>1 hence x_C < x_A) so *NO* need to return areaH3!
    #Along BC: it depends on the relative absissa of B relative to that of C, i.e. it depends on q, k!
    # The Areas computations could use np.trapz for trapezoidal numerical integration approximation.
    # HOWEVER: I found out that the simpson method gives better results, and give good areas values when NPts is high enough: ~8193 points.
    areaH1 = simps(temp1, eNtrpy1)
    #areaH2? >>> Get the index at which eNtropyBC is maximum, then get the temperature at this maximum of Entropy
    maxEntropy = np.max(eNtrpy2)
    iNdX = np.abs(eNtrpy2 -maxEntropy).argmin()
    #print(maxEntropy, iNdX)
    areaH2P = simps(temp2[0:iNdX], eNtrpy2[0:iNdX])
    areaH2M = simps(temp2[iNdX:NPts], eNtrpy2[iNdX:NPts])
    #areaH3 = np.trapz(temp3, eNtrpy3)

    #Shades below curves
#    axs[1, 1].fill_between(vol2, press2, color='#cc0000', alpha=0.3)
#    time.sleep(2)
#    fill_between.remove()

    #print(areaH2P, areaH2M)
    return areaW1, areaW2, areaW3, areaH1, areaH2P, areaH2M

st.cache_data(ttl=1200)
def efficiency(volSMax, initV0, qValue, kValue, gmIG):
    if volSMax >= qValue *initV0:
        Eta = 1 /(1 +(2 *gmIG /(gmIG -1) /(kValue -1)))
    elif volSMax <= initV0:
        Eta = (qValue -1) *(gmIG -1) /2
    else:
        Eta = (gmIG**2 -1) /( (2 * (gmIG +1) /(qValue -1)) +((gmIG *kValue /(kValue -1)) -(1 /(qValue -1)))**2)
    
    #print(volSMax, qValue*initV0, kValue, gmIG)
    return Eta


def disableEnable(etaBtn):
    st.session_state["disabled"] = etaBtn


st.cache_data(ttl=1200, experimental_allow_widgets=True)
def main(initP0, initV0, qValue, kValue, gmIG):
    # st.title('st.title')
    # Place the slider in one of the columns
   
    fig, axs = plt.subplots(3, 2, figsize=(12.0, 11.2))
    
    with st.sidebar:
#        tab1, tab2, tab3, tab4, tab5 = st.tabs(
        tab1, tab2, tab3, tab4 = st.tabs(
#            ['**S**tate', '**T**ables', '**E**fficiency', 'ℹ️nfo', '**E**xit'])
            ['**S**tate', '**T**ables', '**E**fficiency', 'ℹ️nfo'])
        with tab1:
            st.subheader(r'$\scriptsize{\textrm{Input data}}$', divider=True)
            #dataL_column.header("A cat")
            # Add a slider to control the period of the sine wave in a Streamlit app
            # period = st.slider("Period of Sine Wave", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            #    period = dataL_column.slider("Period of Sine Wave", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            # See to format st.number_input: https://docs.streamlit.io/library/api-reference/widgets/st.number_input
            initV0 = st.number_input(
                r'***V*$_0$/ℓ**', min_value=5.0, max_value=200.0, value=100.0, step=5.0, format='%.1f')
            initP0 = st.number_input(
                r'***p*$_0$/Pa**', min_value=5000.0, max_value=100000.0, value=25000.0, step=2500.0, format='%.0f')
            # Below: volume in m³ and temp. in K
            
            initT0 = initP0 * initV0/1000 / Rmol

            #st.markdown("---")
            st.subheader('', divider=True)
            qValue = st.number_input('***q***', min_value=1.05, max_value=20.00,
                                     value=2.500, step=0.025, help=r'C($qV_0,P_0$)', format='%.3f')
            kValue = st.number_input('***k***', min_value=1.075, max_value=25.00,
                                     value=4.000, step=0.025, help=r'B($V_0,kP_0$)', format='%.3f')
            
            #st.subheader(r':blue[$\scriptsize{\textrm{Select *γ* = *C*}}$]', divider=True)
            st.subheader('', divider=True)
            gammaIG = st.select_slider(
                r':blue[Select ***γ = C$\sf{_p}$/C$\mathsf{_V}$***]',
                #label_visibility="hidden", 
                options=['7/5', '5/3'])
            if gammaIG == '7/5':
                gmIG=7/5
            else:
                gmIG=5/3


        with tab2:
            helpTempSonifStr='''***Listen to*** the _temperature variation_  
                                along the :red[BC] process whilst  the  
                                pressure decreases linearly!'''
            st.subheader(r'$\scriptsize{\textrm{Temperature sonification}}$', help=helpTempSonifStr, divider=True)
           # launchSonification = st.checkbox('Launch sonification!', help=helpTempSonifStr)
           # if launchSonification:
                #st.write('Great!')
            volIsobaric, volIsoKoric, volInfnThrm = VolumeEvol(initV0, qValue)
            pIsobaric, pIsoKoric, pInfnThrm = PressureEvol(initP0, kValue)
            temInfnThrm = defTemp(volInfnThrm, pInfnThrm)
            temIsobaric = defTemp(volIsobaric, pIsobaric)
            temIsoKoric = defTemp(volIsoKoric, pIsoKoric)

            sndSplRt = 3277
            durationSec = NPts /sndSplRt
            iTemp=np.linspace(0, NPts -1, NPts)
            timeWave=np.linspace(0, int(durationSec), NPts, False)
          # Fit a quadratic polynomial (degree 2) to the infinitherm data
            fitCoefTempBC = np.polyfit(iTemp, temInfnThrm /initT0, 2)
           #print(fitCoefTempBC)
           #poly = np.poly1d(fitCoefTempBC)
           # Generate points on the fitted curve for smooth plotting
           #x_fit = np.linspace(min(iTemp_BC), max(iTemp_BC), NPts)
           #y_fit = poly(x_fit)
            fModBC = kValue + fitCoefTempBC[1] *(sndSplRt) *(timeWave) + fitCoefTempBC[0] *(sndSplRt ** 2) *(timeWave **2)
            fModAB = 1 + (kValue -1) /(NPts -1) *(sndSplRt) *(timeWave)
            fModCA = qValue - (qValue -1) /(NPts -1) *(sndSplRt) *(timeWave)

            sineSound_BC = np.sin(2*np.pi*(440+20*fModBC) *timeWave)
            sineSound_AB = np.sin(2*np.pi*(440+20*fModAB) *timeWave)
            sineSound_CA = np.sin(2*np.pi*(440+20*fModCA) *timeWave)
          # Save the sound array as a WAV or OGG file
          # sf.write("sndInfnThrm.ogg", np.concatenate((sineSound_AB, sineSound_BC, sineSound_CA)), sndSplRt)
            sf.write("sndInfnThrm.ogg", sineSound_BC, sndSplRt)
          # Display the audio player in Streamlit
            st.audio("sndInfnThrm.ogg", format="audio/ogg")
          # figTS, axTS = plt.subplots(figsize=(6, 4))
          # axTS.plot(timeWave, fModBC, color="#003EFF")
          # st.pyplot(figTS)

          # global volSMax
            #dataL_column.header("A dog")
            st.subheader(r'$\scriptsize{\textrm{Numerical data}}$', divider=True)
            #st.image("https://static.streamlit.io/examples/dog.jpg", width=100)
            #stateTable(initP0, initV0, initT0, qValue, kValue, volSMax)
            tempMax, volTMax, volSMax, SMax, TNMax, STMax = defMaxData(initP0, initV0, qValue, kValue, gmIG)
            #print(tempMax, volTMax, volSMax, SMax, TNMax, STMax)

            tableStates = f"""
                | State  | *p*/kPa | *V*/ℓ | *T*/K |
                |--------|---------|-------|-------|
                | **A**  | {round(initP0/1000,1)}                 | {round(initV0,1)}         | {round(initT0,1)}         |
                | **B**  | {round(kValue *initP0/1000,1)}         | {round(initV0,1)}         | {round(kValue *initT0,1)} |
                | **C**  | {round(initP0/1000,1)}                 | {round(qValue *initV0,1)} | {round(qValue *initT0,1)} |
                | **Z(*T*ₘₐₓ)** | {round(Rmol*tempMax/volTMax,1)} | {round(volTMax,1)}        | {round(tempMax,1)}        |
                | **N(*S*ₘₐₓ)** | {round(Rmol*TNMax/volSMax,1)}   | {round(volSMax,1)}        | {round(TNMax,1)}          |
                """
                #Super- and sub- scripts come from: https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts#Superscript_IPA

            wQuS(initV0, initP0, qValue, kValue, gmIG)
            tableProcesses = f"""
                | Process | *W*/kJ | *Q*/kJ | Δ*U*/kJ | Δ*S*/J/K |
                |---------|--------|--------|---------|----------|
                | **AB**  | {round(ABWork,3)} | {round(ABHeat,3)} | {round(ABintEgy,3)} | {round(ABeNtrpy,2)} |
                | **BC**  | {round(BCWork,3)} | {round(BCHeat,3)} | {round(BCintEgy,3)} | {round(BCeNtrpy,2)} |
                | **CA**  | {round(CAWork,3)} | {round(CAHeat,3)} | {round(CAintEgy,3)} | {round(CAeNtrpy,2)} |
                """

            # Sample DataFrame: another way to build simple table, identical looking
            # data = {
            #     'State': ['**A**', '**B**', '**C**', '**Z**', '**N**'],
            #     '*p*/kPa': [round(initP0/1000,1), round(kValue *initP0/1000,1), round(initP0/1000,1), round(Rmol*tempMax/volTMax,1), round(Rmol*TNMax/volSMax,1)],
            #     '*V*/ℓ': [round(initV0,1), round(initV0,1), round(qValue *initV0,1), round(volTMax,1), round(volSMax,1)],
            #     '*T*/K': [round(initT0,1) , round(kValue *initT0,1), round(qValue *initT0,1), round(tempMax,1) , round(TNMax,1)]
            # }
            # df = pd.DataFrame(data)
            # # Generate Markdown table content dynamically from DataFrame
            # markdown_table = "| " + " | ".join(df.columns) + " |\n"
            # markdown_table += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            # for index, row in df.iterrows():
            #     markdown_table += "| " + " | ".join(str(val) for val in row.values) + " |\n"
            # # Display the generated Markdown table using st.markdown
            # st.markdown(markdown_table)



# stateData = pd.DataFrame(
           # [
           #     {"State": "A", r"$p$/kPa": initP0/1000, "V/ℓ": initV0, "T/K": round(initT0,1)},
           #     {"State": "B", r"$p$/kPa": kValue *initP0/1000, "V/ℓ": initV0, "T/K": round(kValue *initT0,1)},
           #     {"State": "C", r"$p$/kPa": initP0/1000, "V/ℓ": qValue *initV0, "T/K": round(qValue *initT0,1)},
           #     {"State": "N", r"$p$/kPa": np.nan, "V/ℓ": volSMax, "T/K": np.nan},
           # ]
           # )
           # st.dataframe(stateData, use_container_width=False, hide_index=True)
           # st.divider()
            st.subheader(r'$\scriptsize{\textrm{States}}$', divider=True)
            st.markdown(tableStates)
            st.subheader(r'$\scriptsize{\textrm{Processes}}$', divider=True)
            st.markdown(tableProcesses)


        with tab3:
            infoStrg = r'''Compare *η* derived from two methods:  
                       theory *versus* numerical integration.  
                       Efficiency is a dimensionless performance  
                       measure of the cycle. A generic definition  
                       is: *η* = $\frac{benefit}{cost}$.'''
            st.subheader(r'Compute efficiency $\ $ *η* $\quad $', divider=True, help=infoStrg)
            Eta = efficiency(volSMax, initV0, qValue, kValue, gmIG)

#            st.subheader(r'Numerically computed $\ $ *η* $\quad $', divider=True, help='Computations according to formula.')
            mrkDnStr = r'$\hspace{0.65em}$' + 'Calculated *η* ' + r'$=$' +f'{Eta*100:.4f}%' + r'$\hspace{0.35em}$'
            effHlpStr = r'''Value inferred from a formula:  
                        • if *V*$_{\sf{_ɴ}} \ge\ $*qV*₀:$\enspace \cfrac{1}{η}$ = 1 + $\cfrac{2 γ}{(𝘬-1)(γ-1)}$  
                        • if *V*$_{\sf{_ɴ}} \le\ $*V*₀:$\quad$  *η* = ½ (*q*-1) (𝛾-1)   
                        • else:$\quad \qquad \ $  *η* = (𝛾²-1)/*D*₀, with:  
                        $\quad \quad$ *D*₀ = $\sf{\cfrac{2(γ+1)}{𝘲-1}}+\left(\cfrac{γ𝘬}{𝘬-1}-\cfrac{1}{𝘲-1}\right)\raisebox{0.75em}{²}$'''
            st.markdown(mrkDnStr, help=effHlpStr)

#            st.subheader(r'Graphically compute $\ $ *η* $\quad $', divider=True, help='Result based on (*p*,*V*) & (*T*,*S*) area digital computations  \n using trapezoidal method.')
            #if st.button('Compute *η*', help='Click to launch the computation!', on_click=areaWH, args=['Streamlit'], type="secondary", disabled=False, use_container_width=False):
            helpStr='''
                    Compute ***W*** & ***Q₊*** hatched-areas using  
                    the composite trapezoidal rule, a nu-  
                    merical approximation of integrals.'''
            helpStr2='''
                    Clear areas hatched once the theoretical  
                    theoretical & graphical efficiency values  
                    have been checked to be equal.'''
            
            btnPlaceholder = st.empty()
            etaBtn = btnPlaceholder.button(r'*η* $\ $ from ratio of areas', disabled=False, key='etaBtn1', help=helpStr)
#            if st.button(r'*η* $\ $ from ratio of areas', key='etaBtn', disabled=False, help=helpStr):
            if etaBtn:
                btnPlaceholder.button(r'*η* $\ $ from ratio of areas', disabled=True, key='etaBtn2', help=helpStr)
                volIsobaric, volIsoKoric, volInfnThrm = VolumeEvol(initV0, qValue)
                pIsobaric, pIsoKoric, pInfnThrm = PressureEvol(initP0, kValue)
                temIsobaric = defTemp(volIsobaric, pIsobaric)
                temIsoKoric = defTemp(volIsoKoric, pIsoKoric)
                temInfnThrm = defTemp(volInfnThrm, pInfnThrm)
                eNtrpyAB, eNtrpyBC, eNtrpyCA = eNtrpyEvol(initV0, initP0, initT0, qValue, kValue, gmIG)
                areaW1, areaW2, areaW3, areaH1, areaH2P, areaH2M = areaWH(pIsoKoric, pInfnThrm, pIsobaric, volIsoKoric, volInfnThrm, volIsobaric, temIsoKoric, eNtrpyAB, temInfnThrm, eNtrpyBC, temIsobaric, eNtrpyCA)
                #etaEfficiency =r'''*W*$\mathrm{_{tot}}$ = *W*$\mathrm{_{AB}}$ + *W*$\mathrm{_{BC}}$ + *W*$\mathrm{_{CA}}$  
                #            = {areaW1} + {areaW2} + {areaW3}  
                #            = {round(areaW1 +areaW2 +areaW3), 3}
                #'''
                #Below: wrapped between '$', the following non italic LaTeX command work (don't forget '{}'!): \rm{}, \mathrm{}, \sf{}, \mathsf{},
                #       However: \rm and \mathrm{} uses serif font, while the others use sans-serif family.
                #WArea =r'*W*$\sf{_{tot}}$ = *W*$\sf{_{AB}}$ + *W*$\mathsf{_{BC}}$ + *W*$\sf{_{CA}}$  $\\$' #' $\\$ = 23.12'
                #WArea +=r'*W*$\sf{_{tot}}$ = ' + f' {areaW1/1E06:.3f} + {areaW2/1E06:.3f} - {areaW3/1E06:.3f} kJ  \n'
                helpWarea="""It is an engine cycle: the system ***yields***  
                             work, computed as the cycle area."""
                helpSarea="""Only the heat  ***absorbed***  by the  
                            the system has to be counted."""
                WAreaT =r'*W*$\sf{_{tot}} = -$' + f' {(areaW1 +areaW2 -areaW3)/1E06:.4f} kJ'
                st.markdown(WAreaT, help=helpWarea)
                SAreaP =r'*Q*$\sf{_{+}} = $' + f' {(areaH1 +areaH2P) /1E03:.4f} kJ'
                st.markdown(SAreaP, help=helpSarea)
                etaArea = (areaW1 +areaW2 -areaW3) /(areaH1 +areaH2P) /1E03
                ratioStr = r'$\sf{\frac{-Work \ released}{\ \ \ Heat \ received}}$' + f' = {etaArea*100:.4f}%'
                st.markdown(ratioStr)

                areaH2PStr = "{:.4f}".format(np.round(areaH2P))
                areaH2MStr = "{:.4f}".format(-np.round(areaH2M))
                deNomi = (qValue -1) *(gmIG -1)
              # print(areaH2PStr, areaH2MStr)
                if (areaH2PStr == areaH2MStr):
                    #print(areaH2PStr, areaH2MStr)
                    axs[1, 0].text(SMax/1.25, 0.35, r'$Q\sf{^{BC}_{\mathrm{+}}}=|\mathit{Q}\sf{^{BC}_{\mathrm{-}}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))
                    axs[1, 1].text(volSMax*0.700, (qValue *(1 -kValue) +gmIG *(qValue -1)) /deNomi, r'$Q\sf{^{BN}}=|\mathit{Q}\sf{^{NC}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))
                elif (np.round(areaH2P) <np.round(-areaH2M)):
                    axs[1, 0].text(SMax/1.25, 0.35, r'$Q\sf{^{BC}_{\mathrm{+}}}<|\mathit{Q}\sf{^{BC}_{\mathrm{-}}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))
                    axs[1, 1].text(volSMax*0.700, (qValue *(1 -kValue) +gmIG *(qValue -1)) /deNomi, r'$Q\sf{^{BN}}<|\mathit{Q}\sf{^{NC}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))
                else:
                    axs[1, 0].text(SMax/1.25, 0.35, r'$Q\sf{^{BC}_{\mathrm{+}}}>|\mathit{Q}\sf{^{BC}_{\mathrm{-}}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))
                    axs[1, 1].text(volSMax*0.700, (qValue *(1 -kValue) +gmIG *(qValue -1)) /deNomi, r'$Q\sf{^{BN}}>|\mathit{Q}\sf{^{NC}}|$', bbox=dict(facecolor='skyblue', alpha=0.4, edgecolor='red', boxstyle='round'))


                #Shades below curves
                #axs[0, 0].fill_between(volInfnThrm, pInfnThrm/1000, color='#cc0000', alpha=0.3, hatch='/')
                #fill0=axs[0, 0].fill_between(volInfnThrm, pInfnThrm/1000, color='none', alpha=0.3, hatch='//', edgecolor='#AF0733')
                axs[0, 0].fill_between(volInfnThrm, pInfnThrm/1000, color='none', alpha=0.3, hatch='//', edgecolor='#AF0733')
                axs[0, 0].fill_between(volIsobaric, pIsobaric/1000, color='none', alpha=0.3, hatch='\\\\', edgecolor='#003EFF')

                iNdX = np.abs(eNtrpyBC - SMax).argmin()
                axs[1, 0].fill_between(eNtrpyBC[0:iNdX], temInfnThrm[0:iNdX]/initT0, color='none', alpha=0.3, hatch='//', edgecolor='#AF0733')
                #axs[1, 0].fill_between(eNtrpyCA, temIsobaric/initT0, color='none', alpha=0.3, hatch='\\\\', edgecolor='#003EFF')
                axs[1, 0].fill_between(eNtrpyAB, temIsoKoric/initT0, color='none', alpha=0.3, hatch='||', edgecolor='#009900')

                axs[0, 0].text(initV0 *(2.5 +qValue) /4, initP0/1000 *(2.5 +kValue) /4, r'$W\sf{_{\mathrm{tot}}}$$=-Area\sf{_{\mathrm{cycle}}}$')
                axs[1, 0].text(SMax/4, (qValue +kValue)/2.5, r'$Q\sf{_{\mathrm{+}}}$$=-Area\sf{_{\mathrm{cycle}}}$')

                #btnPlaceholder.button(r'*η* $\ $ from ratio of areas', disabled=False, key='etaBtn3', help=helpStr)
                #btnPlaceholder.empty()
                if st.button(r'Clear hatched areas', help=helpStr2):
                    #time.sleep(7.75)
                    st.empty()
                    #fill0.remove()
                    #fill1.remove()
                    #fill2.remove()
                    #fill3.remove()
            

        with tab4:
            #st.subheader(r'$\scriptsize{\textrm{Info}}$', divider=True)
            languageSel = st.checkbox('Fr', value=False)
            # st.divider()
            if languageSel:
                st.markdown(tab4_Aide)
            else:
                st.markdown(tab4_helpContent)

#        with tab5:
#            # Create a button to exit the app
#            if st.button('Thanks for using this app!'):
#                st.stop()

    st.subheader('Triangular reversible engine cycle')
    # Create a figure and axis
    #volIsobaric = np.linspace(initV0, qValue * initV0, NPts)
    #pIsobaric = np.linspace(initP0, initP0, NPts)
    volIsobaric, volIsoKoric, volInfnThrm = VolumeEvol(initV0, qValue)
    pIsobaric, pIsoKoric, pInfnThrm = PressureEvol(initP0, kValue)
    temIsobaric = defTemp(volIsobaric, pIsobaric)
    #volIsoKoric = np.linspace(initV0, initV0, NPts)
    #pIsoKoric = np.linspace(initP0, kValue * initP0, NPts)
    temIsoKoric = defTemp(volIsoKoric, pIsoKoric)
    #volInfnThrm = np.linspace(initV0, qValue * initV0, NPts)
    #pInfnThrm = np.linspace(kValue * initP0, initP0, NPts)
    temInfnThrm = defTemp(volInfnThrm, pInfnThrm)

    # Create a figure containing 4 subplots arranged in 2 rows by 2 columns, with index 1 to 4
    #plt.figure(1, figsize=(6, 4), dpi=300, facecolor=('b'), edgecolor=('red'), frameon='true')
    #st.title('Subplots arranged in a 2 x 2 grid')
    # Subplot #1
    #fig1, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
#    fig, axs = plt.subplots(3, 2, figsize=(12.0, 11.2))
    # Set axis limits to ensure the line segment is visible
    axs[0, 0].set_xlim(initV0 * 0.85, initV0 * qValue * 1.15)
    axs[0, 0].set_ylim(0, kValue *initP0/1000 +20)
    # Get some data for useful information to display on the plot:
    #tempMax, volTMax, volSMax, SMax, TNMax, STMax = defMaxData(initP0, initV0, qValue, kValue, gmIG)
    #print(tempMax, volTMax, volSMax, SMax, TNMax, STMax)
    # Tag for Tmax: finds the index for the infinitherm x/volume value, then retrieve the corresponding y/pressure value
    index = np.abs(volInfnThrm - volTMax).argmin()
    iNdx = np.abs(volInfnThrm - volSMax).argmin()
    # Add labels and a title (optional)
    axs[0, 0].plot(volIsoKoric, pIsoKoric/1000, color="#009900")
    axs[0, 0].plot(volInfnThrm, pInfnThrm/1000, label=r'Infinitherm: $p(V)=-aV+b$'+'\n' +
                   r'$\quad T_{\mathrm{max}}=$'+f'{tempMax:.1f} K'+'\n'+r'$\quad V_{\mathrm{max}}=$'+f'{volTMax:.1f}'+r' ℓ ; '+r'$V_{\mathrm{N}(S_{\mathrm{max}})}=$'+f'{volSMax:.1f}'+r' ℓ', color="#AF0733")
    axs[0, 0].plot(volIsobaric, pIsobaric/1000, color="#003EFF")
    axs[0, 0].set_xlabel('V/ℓ', fontweight='bold', fontstyle='italic')
    axs[0, 0].set_ylabel(r'$\mathbfit{p}\mathbf{/kPa}$', fontweight='bold', rotation=0, va='top', y=0.96, labelpad=18)
    #ax.set_title(r'$p,V$ Diagram', fontweight='bold', fontsize=14, loc='left')
    #ax.set_title("$\mathbf{p,V \mathrm{Diagram}}$", fontweight='bold', fontsize=14, loc='left')
    #ax.set_title("$\mathrm{Roman} \mathit{Italic} \mathbf{bold-Italic}$", fontsize=14, loc='left')
    #ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!', fontsize=14, loc='left')
    axs[0, 0].set_title(r'(p,V) $\mathbf{Diagram}$', fontsize=12, loc='center',
                        fontstyle='italic', fontweight='bold', fontfamily='sans-serif')
    #plt.title("$\mathcal{A}\mathrm{sin}(2 \omega t)$", fontsize=22)
    # ax.legend()
     # Show the plot
    #plt.text(initV0*(1+qValue)/2, initP0*(1+kValue)/2, "infinitherm process")
    axs[0, 0].text(initV0*0.925, initP0/1000*0.725, r'A($V_0,p_0$)')
    axs[0, 0].text(initV0*0.925, initP0/1000*kValue*1.025, r'B($V_0,k p_0$)')
    axs[0, 0].text(initV0*qValue*0.9750, initP0/1000*0.725, r'C($q V_0,p_0$)')
    #    plt.figure(1, figsize=(6, 4), dpi=300, facecolor=('b'), edgecolor=('red'), frameon='true')
    # Highlight the specific point
     # ax.scatter(volTMax, pInfnThrm[index]/1000, marker='|', color='#AF0733', label='$V_{max}=$'+f'{volTMax:.1f}'+' $\ell$')
    # ax.scatter(volTMax, pInfnThrm[index]/1000, marker='|', color='#AF0733')
    # Alternative to highlight the specific point with an arrow
    # ax.annotate(r'$T_{max},'+'\n'+r'$V_{max}=$'+f'{volTMax:.1f}'+r' $\ell$',
    #            xy=(volTMax, pInfnThrm[index]/1000),
    #            xytext=(volTMax + 0.5, pInfnThrm[index]/1000 + 0.5),  # Adjust the text position
    #            arrowprops=dict(facecolor='red', shrink=0.05),
    #            fontsize=10, color='red')
    # See for detail: https://matplotlib.org/stable/gallery/text_labels_and_annotations/annotation_demo.html
    #el = Ellipse((2, -1), 0.5, 0.5)
    # ax.add_patch(el)
    # ax.annotate(r'$T_{\mathrm{max}}=$'+f'{tempMax:.1f} K'+'\n'+r' $V_{\mathrm{max}}=$'+f'{volTMax:.1f}'+r' $\ell$',
    #            xy=(volTMax, pInfnThrm[index]/1000), xycoords='data',
    #            xytext=(volTMax - 20, pInfnThrm[index]/1000 + 15), textcoords='data',
    #            arrowprops=dict(arrowstyle="->", color='#AF0733'),
    #                            #patchB=el,
    #                            #connectionstyle="angle,angleA=90,angleB=0,rad=10"),
    #            fontsize=11, color='#AF0733')
    #plt.text(volTMax, pInfnThrm[index]/1000, r'$T_{max}$'+'\n')
    if (initV0 < volTMax < qValue *initV0):
        axs[0, 0].text(volTMax*1.00, pInfnThrm[index]/1000*1.05, r'Z($T_{\mathrm{max}}$)', color='#AF0733')
        axs[0, 0].scatter(volTMax, pInfnThrm[index]/1000, marker='.', color='#AF0733')
    if (initV0 < volSMax < qValue *initV0):
        axs[0, 0].text(volSMax*1.00, pInfnThrm[iNdx]/1000*1.10, r'N($S_{\mathrm{max}}$)', color='#AF0733', bbox=dict(facecolor='white', alpha=0.7, edgecolor='None'))
        axs[0, 0].scatter(volSMax, pInfnThrm[iNdx]/1000, marker='.', color='#AF0733')
    axs[0, 0].legend()
    # plotR_column.pyplot(fig)

    # Subplot#2
    # ! This plot is segmented to distinguish the temperature actually followed during the infinitherm
    #   process from the temperature computed outside this process (dashed line) !
    #    fig, infnThrmTvsV = plt.subplots()
    # Set axis limits to ensure the line segment is visible
    # Call this function to update volTMax & tempMax
    #tempMax, volTMax, volSMax, SMax, TNMax, STMax = defMaxData(initP0, initV0, qValue, kValue, gmIG)
    axs[0, 1].set_xlim(0, 2*volTMax *1.075)
    #print(initT0, tempMax, tempMax /initT0, tempMax /initT0 *1.35)
    axs[0, 1].set_ylim(0, tempMax /initT0 *1.35)
    axs[0, 1].set_xticks([0, volTMax, volSMax, 2*volTMax], labels=[0, r'$V_{\mathrm{max}}$', r'$V_{\mathrm{N}}$', r'$2V_{\mathrm{max}}$'])
    # Add labels and a title (optional)
    eXtendedVol, eXtendedPre, eXtendedTem = defXtendedData(initP0, initT0, qValue, kValue, volTMax)
    # Split the x values into segments
    segment_points = [0, initV0, qValue*initV0, 2*volTMax]
    for i in range(len(segment_points) - 1):
        start_index = np.searchsorted(eXtendedVol, segment_points[i])
        end_index = np.searchsorted(eXtendedVol, segment_points[i + 1])
        segment_xVol = eXtendedVol[start_index:end_index]
        segment_yTem = eXtendedTem[start_index:end_index]
        # Alternate between dashed and solid lines
        linestyle = '--' if i % 2 == 0 else '-'  # Dashed/Solid for even/odd segments
        # Label for the legend
        label = 'actual path' if i == 1 else 'eXtended range' if i == 2 else ''
        # Plot the segment
        #plt.plot(segment_x, segment_y, label=label, color=color, linestyle=linestyle)
        axs[0, 1].plot(segment_xVol, segment_yTem, label=label, linestyle=linestyle, color="#AF0733")

   # infnThrmTvsV.plot(eXtendedVol, eXtendedTem, color="#AF0733")
   # axs[0, 1].set_xlabel(r'$\mathbfit{V/\ell}$')
    axs[0, 1].set_ylabel(r'$\mathbfit{\dfrac{T}{T\mathbf{_0}}}$', rotation=0, ha='right', y=0.925, labelpad=6)
    axs[0, 1].set_title(r'$\mathbf{Infinitherm}$: $\mathbfit{\frac{T}{T\mathbf{_0}}(V)=\frac{-aV²+bV}{R\ T\mathbf{_0}}}$',
                        fontsize=12, loc='center', fontweight='bold', color='#9F000F')
    axs[0, 1].text(initV0*0.98, kValue*0.90, 'B')
    axs[0, 1].scatter(initV0, kValue, marker='|', color='#AF0733')
    axs[0, 1].text(qValue *initV0*0.975, qValue*0.865, 'C')
    axs[0, 1].scatter(qValue*initV0, qValue, marker='|', color='#AF0733')
    axs[0, 1].text(volSMax*0.965, TNMax/initT0*1.050, r'N($S_{\mathrm{max}}$)')
    axs[0, 1].scatter(volSMax, TNMax/initT0, marker='|', color='#AF0733')
    #axs[0, 1].text(volTMax*0.935, (qValue*kValue -1)**2 /4 /(qValue-1)/(kValue-1)*1.085, r'$\dfrac{T_{\mathrm{max}}}{T_0}$', color='#AF0733')
    #axs[0, 1].text(volTMax*0.935, (qValue*kValue -1)**2 /4 /(qValue-1)/(kValue-1)*1.085, r'${\mathrm{Z}}(\dfrac{T_{\mathrm{max}}}{T_0})$')
    axs[0, 1].text(volTMax*0.975, (qValue*kValue -1)**2 /4 /(qValue-1)/(kValue-1)*1.06, 'Z')
    axs[0, 1].scatter(volTMax, (qValue*kValue - 1)**2 / 4 /(qValue-1)/(kValue-1), marker='|', color='#AF0733')
    axs[0, 1].legend(loc='upper right')
    # Show the plot
    # plotR_column.pyplot(fig)


    # Subplot#3
    eNtrpyAB, eNtrpyBC, eNtrpyCA = eNtrpyEvol(initV0, initP0, initT0, qValue, kValue, gmIG)
    #    fig, TvsS = plt.subplots()
    axs[1, 0].set_xlim(-1.5, SMax +5.5)
    axs[1, 0].set_ylim(0, (tempMax /initT0) +1.0)
    axs[1, 0].plot(eNtrpyAB, temIsoKoric /initT0, label='Isochoric', color="#009900")
    axs[1, 0].plot(eNtrpyBC, temInfnThrm /initT0, label='Infinitherm', color="#AF0733")
    axs[1, 0].plot(eNtrpyCA, temIsobaric /initT0, label='Isobaric', color="#003EFF")
    axs[1, 0].set_xlabel(r'$\mathbfit{S} \mathbf{/J/K}$')
    axs[1, 0].set_ylabel(r'$\mathbfit{\dfrac{T}{T\mathbf{_0}}}$', rotation=0, ha='right', y=0.925, labelpad=6)
    axs[1, 0].set_title(r'$\mathbfit{(\frac{T}{T\mathbf{_0}}\mathbfit{,S)}} \ \mathbf{Diagram}$', fontsize=13, loc='center')
    axs[1, 0].text(0-.500, 1*1.095, 'A')
    axs[1, 0].text(eNtrpyBC[0]*0.965, kValue*1.025, 'B')
    axs[1, 0].text(eNtrpyBC[NPts-1]*1.010, qValue*.8250, 'C', va='bottom', ha='right')
    #if ((tempMax/initT0 > kValue) or (tempMax/initT0 > qValue)):
    if (initV0 < volTMax < qValue *initV0):
        axs[1, 0].text(STMax*0.96, tempMax/initT0*1.075, r'$\dfrac{T_{\mathrm{max}}}{T_0}$', color='#AF0733')
        axs[1, 0].scatter(STMax, tempMax/initT0, marker='.', color='#AF0733')
    #print(eNtrpyBC[0], SMax, eNtrpyBC[NPts -1])
    #if ((SMax > eNtrpyBC[0]) or (SMax > eNtrpyBC[NPts -1])):
    if (initV0 < volSMax < qValue *initV0):
        axs[1, 0].text(SMax*1.015, TNMax/initT0*1.00, r'$S_{\mathrm{max}}$(N)', color='#AF0733')
        axs[1, 0].scatter(SMax, TNMax/initT0, marker='.', color='#AF0733')
    axs[1, 0].legend(loc='upper left')
    
    # Make the shaded regions according to 2 methods, 1st one (using Polygon & add.patch) seems slower
    # Get the index at which eNtropyBC is maximum, then get the temperature at this maximum of Entropy
    iNdX = np.abs(eNtrpyBC - SMax).argmin()
#    S_shade = eNtrpyBC[0:iNdX]
#    T_shade = temInfnThrm[0:iNdX] /initT0
#    verts = [(eNtrpyBC[0], 0), *zip(S_shade, T_shade), (eNtrpyBC[iNdX], 0)]
#    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
#    axs[1, 0].add_patch(poly)
    S_shadeP = eNtrpyBC[0:iNdX]  # Specify the x range for shading
    T_shadeP = temInfnThrm[0:iNdX] /initT0  # Corresponding y values within the specified x range
    #See about colors: https://matplotlib.org/stable/gallery/color/color_demo.html#sphx-glr-gallery-color-color-demo-py
    axs[1, 0].fill_between(S_shadeP, T_shadeP, color='#cc0000', alpha=0.3)
    S_shadeM = eNtrpyBC[iNdX +1:NPts -1]  # Specify the x range for shading
    T_shadeM = temInfnThrm[iNdX +1:NPts -1] /initT0  # Corresponding y values within the specified x range
    axs[1, 0].fill_between(S_shadeM, T_shadeM, color='skyblue', alpha=0.3)


    # Subplot#4
    #    fig, PvsV = plt.subplots()
    deNomi = (qValue -1) *(gmIG -1)
    qPfunc =(qValue *kValue -1) *gmIG /deNomi -(kValue -1) *(gmIG +1) /deNomi *eXtendedVol/initV0
   #print(qPfunc[0], qPfunc[2 *NPts -2])
   #print(initP0)
    axs[1, 1].sharex(axs[0, 1])
    #axs[1, 1].set_xlim(0, 2*volTMax *1.075)
    #print(initT0, tempMax, tempMax /initT0, tempMax /initT0 *1.35)
    axs[1, 1].set_ylim(qPfunc[2 *NPts -2] *1.10, qPfunc[0] *1.20)
    #axs[1, 1].spines['bottom'].set_position(('data', 0))
    #axs[1, 1].spines['bottom'].set_visible(True)
   #axs[1, 1].set_xticks([0, volTMax, volSMax, 2*volTMax], labels=[0, r'$V_{\mathrm{max}}$', r'$V_{\mathrm{N}}$', r'$2V_{\mathrm{max}}$'])
    axs[1, 1].set_yticks([(qValue *kValue -1) *gmIG /deNomi, (qValue *kValue -1) /2 /(qValue -1), 0, (1 -qValue *kValue) /deNomi], labels=[r'$\frac{γ}{γ-1}\frac{b}{P_0}$', r'$\frac{b}{2P_0}$', 0, r'$\frac{-1}{γ-1}\frac{b}{P_0}$'])
    axs[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Horizontal line at y=0
    #axs[1, 1].axhline(y=(1 -qValue *kValue) /deNomi, color='black', linestyle='--', linewidth=0.8)  # Horizontal line at y=-b/(gamma-1)/P0
    # Add labels and a title (optional)
    # Split the x values into segments
#    segment_points = [0, initV0, qValue*initV0, 2*volTMax]
    for i in range(len(segment_points) - 1):
        start_index = np.searchsorted(eXtendedVol, segment_points[i])
        end_index = np.searchsorted(eXtendedVol, segment_points[i + 1])
        segment_xVol = eXtendedVol[start_index:end_index]
        segment_yPre = qPfunc[start_index:end_index]
        # Alternate between dashed and solid lines
        linestyle = '--' if i % 2 == 0 else '-'  # Dashed/Solid for even/odd segments
        # Label for the legend
#       label = 'actual path' if i == 1 else 'eXtended range' if i == 2 else ''
        # Plot the segment
        #plt.plot(segment_x, segment_y, label=label, color=color, linestyle=linestyle)
        axs[1, 1].plot(segment_xVol, segment_yPre, label=label, linestyle=linestyle, color="#AF0733")

    # infnThrmTvsV.plot(eXtendedVol, eXtendedTem, color="#AF0733")
    axs[1, 1].set_xlabel('V/ℓ', fontweight='bold', fontstyle='italic')
    axs[1, 1].set_ylabel(r'$\mathbfit{f(V)}$', rotation=0, ha='right', y=0.975, labelpad=-15)
    axs[1, 1].set_title(r'$\mathbf{Sign \ of \ }$ $\mathbfit{δQ_{BC} \ ∝ \ } \mathbfit{f(V)=-\frac{(γ+1)}{(γ-1)} \ ^{a} \frac{V}{P\mathbf{_0}} + \frac{γ}{(γ-1)} \ \frac{b}{P\mathbf{_0}}}$',
                        fontsize=12, loc='center', fontweight='bold', color='#9F000F')
    axs[1, 1].text(initV0*0.925, ((1 -kValue) +gmIG * kValue *(qValue -1)) /deNomi *0.685, 'B')
    axs[1, 1].scatter(initV0, ((1 -kValue) +gmIG * kValue *(qValue -1)) /deNomi, marker='|', color='#AF0733')
    axs[1, 1].text(qValue *initV0*0.980, (qValue *(1 -kValue) +gmIG *(qValue -1)) /deNomi *1.300, 'C')
    axs[1, 1].scatter(qValue*initV0, (qValue *(1 -kValue) +gmIG *(qValue -1)) /deNomi, marker='|', color='#AF0733')
    axs[1, 1].text(volSMax*0.975, 0+1.75, r'N($S_{\mathrm{max}}$)')
    axs[1, 1].scatter(volSMax, 0, marker='|', color='#AF0733')
    axs[1, 1].text(volTMax*0.975, (qValue *kValue -1) *(gmIG -1) /2 /deNomi*1.650, 'Z')
    axs[1, 1].scatter(volTMax, (qValue *kValue -1) *(gmIG -1) /2 /deNomi, marker='|', color='#AF0733')
    #axs[1, 1].legend(loc='upper right')

    # Make the shaded regions
    # Get the index at which eXtendedVol is nul, then get qPfunc[0]
    iNdXVN = np.abs(eXtendedVol -volSMax).argmin()
    iNdXVB = np.abs(eXtendedVol -initV0).argmin()
    iNdXVC = np.abs(eXtendedVol -qValue *initV0).argmin()
   #print(iNdXVN,iNdXVB,iNdXVC)
    V_shadeP = eXtendedVol[iNdXVB:iNdXVN]  # Specify the x range for shading
    p_shadeP = qPfunc[iNdXVB:iNdXVN]       # Corresponding y values within the specified x range
    #See about colors: https://matplotlib.org/stable/gallery/color/color_demo.html#sphx-glr-gallery-color-color-demo-py
    axs[1, 1].fill_between(V_shadeP, p_shadeP, color='#cc0000', alpha=0.3)
    V_shadeM = eXtendedVol[iNdXVN +1:iNdXVC]  # Specify the x range for shading
    p_shadeM = qPfunc[iNdXVN +1:iNdXVC]       # Corresponding y values within the specified x range
    axs[1, 1].fill_between(V_shadeM, p_shadeM, color='skyblue', alpha=0.3)

    #Add annotation
    axs[1, 1].annotate(r'$\mathbfit{NB:} \ ΔS_\mathrm{BC} = \dfrac{R}{γ-1} \ Ln\dfrac{q^γ}{k} \quad \quad ΔS_\mathrm{BC} = 0$ if $k=q^γ$', (0.66, 0.98), xycoords='axes fraction', va='top', ha='center')
    axs[1, 1].annotate(r'$e.g.:\ q=2.00, k=2.64, γ=7/5$', (0.66, 0.82), xycoords='axes fraction', va='top', ha='center')

    # Subplot#5 (T/To, v)
    axs[2, 0].sharex(axs[0, 0])
    axs[2, 0].sharey(axs[1, 0])
    axs[2, 0].plot(volIsoKoric, temIsoKoric /initT0, label='Isochoric', color="#009900")
    axs[2, 0].plot(volInfnThrm, temInfnThrm /initT0, label='Infinitherm', color="#AF0733")
    axs[2, 0].plot(volIsobaric, temIsobaric /initT0, label='Isobaric', color="#003EFF")
    axs[2, 0].set_xlabel('V/ℓ', fontweight='bold', fontstyle='italic')
    axs[2, 0].set_ylabel(r'$\mathbfit{\dfrac{T}{T\mathbf{_0}}}$', rotation=0, ha='right', y=0.925, labelpad=6)
    axs[2, 0].set_title(r'$\mathbfit{(\frac{T}{T\mathbf{_0}}\mathbfit{,V)}} \ \mathbf{Diagram}$', fontsize=13, loc='center')

    axs[2, 1].axis('off')

    # Show the plot
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)




if __name__ == "__main__":

    main(initP0, initV0, qValue, kValue, gmIG)
