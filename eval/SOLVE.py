# FUNCTION - CALL XFOIL AND GET AIRFOIL DATA
# Written by: JoshTheEngineer
# YouTube   : www.youtube.com/joshtheengineer
# Website   : www.joshtheengineer.com
# Started   : 02/17/20 - Transferred from MATLAB to Python
#                      - Works as expected
#
# PUROSE
# - Create or load airfoil based on flagAirfoil
# - Save and read airfoil coordinates
# - Save and read airfoil pressure coefficient
# - Save and read airfoil lift, drag, and moment coefficients
#
# INPUTS
# - NACA         : Four-digit NACA airfoil designation
# - PPAR         : Paneling variables used in XFOIL PPAR menu
# - AoA          : Angle of attack [deg]
# - flagAirfoil  : Flag for loading/creating airfoil
# 
# OUTPUTS
# - xFoilResults : Array containing all results

import os  
import time                                                                     # For calling the XFoil executable
import numpy as np                                                              # For number/math stuff
from tkinter import Tk                                                          # For input file dialog box
from tkinter.filedialog import askopenfilename                                  # For input file dialog box
import ntpath                                                                # For file path splitting

def XFOILmod(PPAR,AoA,Ma,Re,path,runName):

    # %% CALL XFOIL FROM MATLAB
    
    xFoilResults = list(range(9))                                               # Initialize results array
    

                                                   # If the user wants to load a DAT file airfoil
    # root = Tk()                                                             # Create GUI for open file dialog box
    # ftypes = [('dat file',"*.dat")]                                         # File types allowed to be loaded
    # ttl  = "Select Airfoil File"                                            # Title of the dialog box GUI
    # dir1 = '/Airfoil_DAT_Selig/'                                            # Initial directory of the dialog box GUI
    # root.withdraw()                                                         # Needed for closing the Tk window later
    # root.update()                                                           # Needed for closing the Tk window later
    # root.fileName = askopenfilename(filetypes = ftypes,                     # User input of airfoil file to load
                                    # initialdir = dir1,
                                    # title = ttl)
    # root.destroy()                                                          # Destroy the Tk window
    
    head, tail = ntpath.split(path)                                         # https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    airfoilName = tail[0:len(tail)-4]                                       # Retain only airfoil name, not extension
    print(airfoilName)
    xFoilResults[0] = airfoilName                                           # Send the airfoil name back from this function
        
    # Save-to file names
    # def correctPath(p):
    #     new_p = p.replace(os.sep, '/')
    #     return new_p
    saveFlnm    = f'./{runName}/tempFiles/Save_' + airfoilName + '.txt'                                # Airfoil coordinates save-to file
    saveFlnmCp  = f'./{runName}/tempFiles/Save_' + airfoilName + '_Cp.txt'                             # Airfoil Cp save-to file
    saveFlnmPol = f'./{runName}/tempFiles/Save_' + airfoilName + '_Pol.txt'                            # Airfoil polar save-to file
    print('nm file:', saveFlnm)
    print('nmcp file:', saveFlnmCp)
    print('nmpol file:', saveFlnmPol)
    # Delete files if they exist
    if os.path.exists(saveFlnm):                                                # If the airofil coordinates file exists
        os.remove(saveFlnm)                                                     # Delete the file
    if os.path.exists(saveFlnmCp):                                              # If the airfoil Cp file exists
        os.remove(saveFlnmCp)                                                   # Delete the file
    if os.path.exists(saveFlnmPol):                                             # If the airfoil polar file exists
        os.remove(saveFlnmPol)                                                  # Delete the file
           
    # Create the airfoil
    input_file = f'./{runName}/tempFiles/xfoil_input.inp'
    fid = open(input_file,"w")                                           # Open a file for writing the XFoil commands to
    fid.write('PLOP\n')
    fid.write('G\n\n')
                                                                         # If the user wants to load an external airfoil file
    print('TRYING TO LOAD:', "LOAD " + f'./{runName}/finGeoms/{tail}\n')
    fid.write("LOAD " + f'./{runName}/finGeoms/{tail}\n')               # Load the airfoil file
        
    fid.write("PPAR\n")                                                         # Enter the PPAR (paneling) menu
    fid.write("N " + PPAR[0] + "\n")                                            # Define "Number of panel nodes"
    fid.write("P " + PPAR[1] + "\n")                                            # Define "Panel bunching paramter"
    fid.write("T " + PPAR[2] + "\n")                                            # Define "TE/LE panel density ratios"
    fid.write("R " + PPAR[3] + "\n")                                            # Define "Refined area/LE panel density ratio"
    fid.write("XT " + PPAR[4] + "\n")                                           # Define "Top side refined area x/c limits"
    fid.write("XB " + PPAR[5] + "\n")                                           # Define "Bottom side refined area x/c limits"
    fid.write("\n")                                                             # Apply all changes
    fid.write("\n")                                                             # Back out to XFOIL menu
    
    # Save the airfoil data points
    fid.write("PSAV " + saveFlnm + "\n")                                        # Save the airfoil coordinate file
    
    # Get Cp and polar data
    fid.write("OPER\n")
    fid.write(f"visc {Re}\n")
    fid.write(f"m {Ma}\n")                                                         # Enter OPER menu
    fid.write("Pacc 1 \n")                                                      # Begin polar accumulation
    fid.write("\n\n")                                                           # Don't enter save or dump file names
    fid.write("Alfa " + str(AoA) + "\n")                                        # Set angle of attack
    fid.write("Alfa " + str(AoA) + "\n")
    fid.write("Alfa " + str(AoA) + "\n")
    fid.write("Alfa " + str(AoA) + "\n")
    fid.write("Alfa " + str(AoA) + "\n")
    fid.write("CPWR " + saveFlnmCp + "\n")                                      # Write the Cp file
    fid.write("PWRT\n")                                                         # Save the polar data
    fid.write(saveFlnmPol + "\n")                                               # Save polar data to this file
    if os.path.exists(saveFlnmPol):                                             # If saveFlnmPol already exists
        fid.write("y \n")                                                       # Overwrite existing file
    # TRY ITERATING FOR A BIT IF NO RESULTS
    # dataBufferPol = np.loadtxt(saveFlnmPol, skiprows=12) 
    # try:
    #     a = dataBufferPol[1]
    #     b = dataBufferPol[2]
    # except:
    #     for i in range(10):
    #         fid.write("Alfa " + str(AoA) + "\n")
            
    # fid.write("CPWR " + saveFlnmCp + "\n")                                      # Write the Cp file
    # fid.write("PWRT\n")                                                         # Save the polar data
    # fid.write(saveFlnmPol + "\n")                                               # Save polar data to this file
    # if os.path.exists(saveFlnmPol):                                             # If saveFlnmPol already exists
    #     fid.write("y \n")                                                       # Overwrite existing file

    fid.close()                                                                 # Close the input file
    
    # Run the XFoil calling command
    os.system(f"xfoil.exe < {input_file}")                                    # Run XFoil with the input file just created
    
    # Delete file after running
    if os.path.exists(input_file):                                       # If the input file exists
        os.remove(input_file)                                            # Delete the file since we don't need it anymore
    
    # %% READ CP DATA
    # Load the data from the text file
    try:
        print('Trying to load CP file: ', saveFlnmCp)
        dataBufferCp = np.loadtxt(saveFlnmCp, skiprows=3)                           # Read the X, Y, and Cp data from data file
    except:
        print('Could not load CP file, isCrash')
        return [], True
    
    # Extract data from the loaded dataBuffer array   
    xFoilResults[1] = dataBufferCp[:,0]                                         # X-data points
    xFoilResults[2] = dataBufferCp[:,1]                                         # Y-data points
    xFoilResults[3] = dataBufferCp[:,2]                                         # Cp data
    
    # Delete file after loading
    if os.path.exists(saveFlnmCp):                                              # If filename exists
        os.remove(saveFlnmCp)                                                   # Delete the file
    
    # %% READ AIRFOIL COORDINATES
    
    # Load the data from the text file
    dataBuffer = np.loadtxt(saveFlnm, skiprows=0)                               # Read the XB and YB data from the data file
    
    # Extract data from the loaded dataBuffer array
    xFoilResults[4] = dataBuffer[:,0]                                           # Boundary point X-coordinate
    xFoilResults[5] = dataBuffer[:,1]                                           # Boundary point Y-coordinate

    # Delete file after loading
    if os.path.exists(saveFlnm):                                                # If filename exists
        os.remove(saveFlnm)                                                     # Delete the file
    
    # %% READ POLAR DATA
    
    # Load the data from the text file
    dataBufferPol = np.loadtxt(saveFlnmPol, skiprows=12)                        # Read the CL, CD, and CM data from the data file
    print('OK YEET', dataBufferPol)
    # Extract data from the loaded dataBuffer array
    try:
        dataBufferPol = dataBufferPol[-1]
        xFoilResults[6] = dataBufferPol[1] # CL
        xFoilResults[7] = dataBufferPol[2] # CD
        xFoilResults[8] = dataBufferPol[4] # CM
    except:
        print('Could not converge, but no crash')
        xFoilResults[6] = 0
        xFoilResults[7] = 0
        xFoilResults[8] = 0

        return [], True
        
    # Delete file after loading
    if os.path.exists(saveFlnmPol):                                             # If filename exists
        os.remove(saveFlnmPol)                                                  # Delete the file
    
    return xFoilResults, False                                                       # Return the important information from this function







    

