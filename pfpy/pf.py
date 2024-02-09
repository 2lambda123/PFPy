import os
import powerfactory
import numpy as np
import pandas as pd
import json
import csv
import re
from collections import defaultdict
import time

pfhandle = None

class BasePFObject: 
    
    def __init__(self, pfobject):
        """"This function initializes an object with a given pfobject and assigns it to the self.pfobject attribute."
        Parameters:
            - pfobject (object): The pfobject to be assigned to the self.pfobject attribute.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Assigns the given pfobject to the self.pfobject attribute."""
        
        self.pfobject = pfobject

    def __eq__(self, other):
        """"Checks if the current object is equal to another object based on their names."
        Parameters:
            - other (BasePFObject): The object to compare with.
        Returns:
            - bool: True if the names of both objects are equal, False otherwise.
        Processing Logic:
            - Check if the other object is an instance of BasePFObject.
            - If yes, compare the names of both objects.
            - If no, compare the name of the current object with the other object directly."""
        
        if isinstance(other, BasePFObject):
            return self.name == other.name
        return self.name == other

    def __getattr__(self, name):
        """If the item is not found in the main object
        look for it in the Python object"""
        return getattr(self.pfobject, name)
        
    def __setattr__(self, name, value):
        """Sets an attribute on the provided object or its underlying PFObject.
        Parameters:
            - self (BasePFObject): The object to set the attribute on.
            - name (str): The name of the attribute to set.
            - value (any): The value to set the attribute to.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - If the name is 'pfobject', set the attribute directly on the object.
            - If the name is an existing attribute on the underlying PFObject, set the attribute on the PFObject.
            - Otherwise, use the default __setattr__ function to set the attribute."""
        
        if name == 'pfobject':
            self.__dict__[name] = value
        elif hasattr(self.pfobject, name):
            setattr(self.pfobject, name, value)
        else:
            super(BasePFObject, self).__setattr__(name, value)

    @property
    def name(self):
        """Returns the name and class of an object.
        Parameters:
            - self (object): The object to be evaluated.
        Returns:
            - str: The name and class of the object.
        Processing Logic:
            - Concatenates the name and class of the object.
            - Uses the f-string method.
            - Returns the result as a string."""
        
        return f'{self.obj_name}.{self.obj_class}' 
    
    @property
    def obj_name(self):
        """"Returns the location name of the given object.
        Parameters:
            - self (object): The object whose location name is being retrieved.
        Returns:
            - str: The location name of the object.
        Processing Logic:
            - Retrieves the location name from the object's pfobject attribute.""""
        
        return self.pfobject.loc_name

    @property
    def obj_class(self):
        """Returns the class name of the given object.
        Parameters:
            - self (object): The object whose class name is to be returned.
        Returns:
            - str: The class name of the given object.
        Processing Logic:
            - Uses the GetClassName() method.
            - Returns a string.
            - No parameters are required."""
        
        return self.pfobject.GetClassName()

class SubscribablePFObject(BasePFObject):
    
    def __init__(self, pfobject, **kwargs):
        """Initializes the object with a given pfobject and optional observers.
        Parameters:
            - pfobject (object): The pfobject to be used for initialization.
            - **kwargs (optional): Additional keyword arguments that can be passed in, including 'observers' to add observers.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Initializes the object with the given pfobject.
            - Adds any optional observers if provided.
            - If no observers are provided, the function will still run without any errors."""
        
        super().__init__(pfobject)
        self._observers = []
        try:
            self.observers_add(kwargs['observers'])
        except KeyError:
            pass

    def observers_notify(self, event):
        """Notifies all observers of an event.
        Parameters:
            - self (object): The object calling the function.
            - event (str): The event to be passed to the observers.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Loop through all observers.
            - Call the update method on each observer.
            - Pass in the object and event as arguments."""
        
        for observer in self._observers:
            observer.update(self, event)
            
    def observers_add(self, observers):
        """Adds one or more observers to the list of observers.
        Parameters:
            - observers (list or object): List of observers or single observer to be added.
        Returns:
            - None: Does not return anything.
        Processing Logic:
            - Checks if the parameter is a list.
            - If not, converts it to a list.
            - Appends each observer to the list of observers."""
        
        if not isinstance(observers, list):
            observers = [observers]
        for observer in observers:
            self._observers.append(observer)
    
    def observers_remove(self, observers):
        """"Removes given observers from the list of observers for the current object."
        Parameters:
            - observers (list or object): List of observers or single observer to be removed.
        Returns:
            - None: No return value.
        Processing Logic:
            - Converts single observer to list if needed.
            - Iterates through list of observers.
            - Removes each observer from the list of observers."""
        
        if not isinstance(observers, list):
            observers = [observers]
        for observer in observers[0]:
            self._observers.remove(observer)        
        
class Project(BasePFObject):
    
    def __init__(self, project_name, project_folder=None):
        """This function initializes a project by creating a study case container and a networks container.
        It takes in two parameters, project_name and project_folder, where project_folder is an optional parameter.
        It returns an object of type Network.
        The function first tries to join the project_folder and project_name together to create a project_path.
        If this fails, it sets project_path to project_name.
        Then, it activates the project_path and creates the study case and networks containers.
        An example of usage would be:
        project = __init__(project_name='MyProject', project_folder='C:/Users/Projects')"""
        
        try:
            project_path = os.path.join(project_folder, project_name)
        except TypeError:
            project_path = project_name       
        super().__init__(self.activate(project_path))
        # Creating the study case container
        
        self.study_cases = StudyCaseContainer()
        # Creating the networks container
        self.networks = {pfobject.loc_name : Network(pfobject, observers = self.study_cases) 
                        for pfobject in pfhandle.GetProjectFolder('netdat').GetChildren(1, '*.ElmNet', 1)
                        }
        for pfobject in pfhandle.GetProjectFolder('netdat').GetChildren(1, '*.ElmNet', 1):
            self.networks[pfobject.loc_name] = Network(pfobject, observers = self.study_cases)
    
    def activate(self, project_path):
        """Activates a project and returns the active project.
        Parameters:
            - project_path (str): Path to the project to be activated.
        Returns:
            - Project (str): The active project.
        Processing Logic:
            - Calls pfhandle.ActivateProject() to activate the project.
            - If the project cannot be activated, raises a RuntimeError.
            - Otherwise, returns the active project using pfhandle.GetActiveProject().
        Example:
            project = activate("C:/Users/JohnDoe/Documents/Project1")
            print(project)
            # Output: "Project1""""
        
        if pfhandle.ActivateProject(project_path):
            raise RuntimeError('Could not activate the project')
        else:
            return pfhandle.GetActiveProject()
        
class StudyCaseContainer(dict):
    def __init__(self):
        """"Initializes a StudyCase object with a dictionary of StudyCases and sets the active key to the currently active StudyCase."
        Parameters:
            - None
        Returns:
            - None
        Processing Logic:
            - Creates a dictionary of StudyCases.
            - Sets the active key to the currently active StudyCase."""
        
        scs = {}
        obj_list = pfhandle.GetProjectFolder('study').GetChildren(1,'*.IntCase',1)
        for obj in obj_list:
            scs[obj.loc_name] = StudyCase(obj, observers=self)
        super().__init__(scs)
        self._active_key = pfhandle.GetActiveStudyCase().loc_name

    @property
    def active_study_case(self):
        """"Returns the active study case, if one exists, from the current object.
        Parameters:
            - self (object): The current object.
        Returns:
            - object: The active study case, if one exists. Otherwise, returns None.
        Processing Logic:
            - Checks if there is an active key.
            - If there is an active key, returns the corresponding study case.
            - If there is no active key, returns None.""""
        
        if self._active_key:
            return self[self._active_key]
        else:
            return None
    
    def update(self, caller, event):
        """"""
        
        if isinstance(caller, StudyCase):
            if event == 'activated':
                self._active_key = caller.obj_name
            elif event == 'deactivated':
                self._active_key = ''
            else:
                raise ValueError('Unknown update event.')
        elif isinstance(caller, Network):
            if event == 'activated':
                self.active_study_case.networks.add(caller.obj_name)
            if event == 'deactivated':
                self.active_study_case.networks.discard(caller.obj_name)
        else:
            raise ValueError('Unkown caller!')
        
class StudyCase(SubscribablePFObject):

    def __init__(self, pfobject, *args, **kwargs):
        """"""
        
        super().__init__(pfobject)
        try:
            self.observers_add(kwargs['observers'])
        except KeyError:
            pass
        summary = self.GetContents('Summary Grid.ElmNet')[0]
        self.networks = set(ref.obj_id.loc_name for ref in summary.GetChildren(1))
        self.elmres = generate_object_dict(
            pfobject.GetContents('*.ElmRes'), 'ElmRes')
        self.comres = generate_object_dict(
            pfobject.GetContents('*.ComRes'), 'ComRes')
        
        try:
            self.inc = self.GetContents('*.ComInc')[0]
        except IndexError:
            self.inc = self.CreateObject('ComInc', 'Initial Conditions')
        try:
            self.sim = self.GetContents('*.ComSim')[0]
        except IndexError:
            self.sim = self.CreateObject('ComSim', 'Simulation')
        try:
            self.mod = self.GetContents('*.ComMod')[0]
        except IndexError:
            self.mod = self.CreateObject('ComMod', 'Modal Analysis')
        
    def activate(self):
        """"""
        
        if self.GetFullName() == pfhandle.GetActiveStudyCase().GetFullName():
            return True
        elif self.pfobject.Activate():
            raise RuntimeError('Could not activate the study case!')
        else:
            self.observers_notify('activated')
            return True

    def deactivate(self):
        """"""
        
        if self.GetFullName() != pfhandle.GetActiveStudyCase().GetFullName():
            return True
        elif self.pfobject.Deactivate():
            raise RuntimeError('Could not activate the study case!')
        else:
            self.observers_notify('deactivated')
            return True
            
    def _initialize_modal(self,  path, iLeft=1, iRight=1, iPart=1, iSysMatsMatl=1, iEValMatl=1, iREVMatl=1, 
                    iLEVMatl=1, iPartMatl=1):
        """"""
        
        self.mod.iLeft = iLeft
        self.mod.iRight = iRight
        self.mod.iPart = iPart
        self.mod.iSysMatsMatl = iEValMatl
        self.mod.iREVMatl = iREVMatl
        self.mod.iLEVMatl = iLEVMatl
        self.mod.iPartMatl = iPartMatl
        self.mod.dirMatl = path
        
        self.pInitCond = self.inc
        return self.inc.Execute()

    def _initialize_dynamic(self, sim_type, start_time=0.0, step_size=0.01, 
                  end_time=10.0):
        """
        Initialize the dynamic simulation.
        """
        # set start time, step size and end time
        self.inc.tstart = start_time
        self.inc.dtgrd = step_size
        self.sim.tstop = end_time
        # set initial conditions
        return self.inc.Execute()

    def modal_analysis(self, path):
        """"""
        
        pfhandle.ResetCalculation()
        if self._initialize_modal(path):
            raise RuntimeError('Simulation initialization failed.')
        if self.mod.Execute():
            raise RuntimeError('Modal computation failed!')

    def simulate(self, start_time, step_size, end_time,  sim_type='rms'):
        """Initializes and runs a dynamic simulation.
        
        Parameters
        ----------
        sim_type : str {'rms', 'emt'}
            Simulation type which can be either RMS or EMT simulation.
        start_time : float
            Starting time of the simulation.
        step_size : float
            Simulation step time.
        end_time : float
            End time of the simulation.
        """
        pfhandle.ResetCalculation()
        if self._initialize_dynamic(sim_type, start_time, step_size, end_time):
            raise RuntimeError('Simulation initialization failed.')
        if self.sim.Execute():
            raise RuntimeError('Simulation execution failed!')

    def create_elmres(self, elmres_name):
        """Creates an ElmRes object. If there is an existing ElmRes 
        object with the same name, it is overwritten by default.
        
        Parameters
        ----------
        elmres_name : str
            The name of the ElmRes object which is created.
        """
        if (elmres := ElmRes(self.CreateObject('ElmRes',elmres_name))) is None:
            raise Exception('Could not create ElmRes!')
        else:
            self.elmres[elmres.obj_name] = elmres
        return elmres
    
    def create_comres(self, comres_name):
        if (comres := ComRes(self.CreateObject('ComRes', comres_name))) is None:
            raise Exception('Could not create ComRes!')
        else:
            self.comres[comres.obj_name] = comres
        return comres
            
class Network(SubscribablePFObject):
    def __init__(self, pfobject, *args, **kwargs):
        super().__init__(pfobject)
        try:
            self.observers_add(kwargs['observers'])
        except KeyError:
            pass
        # Collecting all IntMat objects
        self.intmats = {}
        for intmat in self.GetContents('*.IntMat', 1):
            self.intmats[intmat.loc_name] = IntMat(intmat)
        # Collecting all ElmFile objects
        self.elmfiles = {}
        for elmfile in self.GetContents('*.ElmFile', 1):
            self.elmfiles[elmfile.loc_name] = ElmFile(elmfile)
        
    def activate(self):
        self.pfobject.Activate()
        self.observers_notify('activated')
        
    def deactivate(self):
        self.pfobject.Deactivate()
        self.observers_notify('deactivated')

class IntMat(BasePFObject):
    
    @property
    def signal(self):
        pass #### NOT FINISHED
    
    @signal.setter
    def signal(self, values):
        nrows = values['time'].shape[0]
        self.Init(nrows, 2)
        for rownr in range(1,nrows+1):
            self.Set(rownr, 1, values['time'][rownr-1])
            self.Set(rownr, 2, values['y1'][rownr-1])
        self.Save()

class ElmFile(BasePFObject):
    
    def create_file(self, signal, filepath):
        """
        Writes a signal to a file that can be read by ElmFile in PowerFactory
        """
        sigmat = np.empty((signal['time'].shape[0],len(signal))) # Allocate a matrix of size (len(time)) x (number of signals)
        sigmat[:,0] = signal['time'] # Write the time vector
        for signum in range(1,len(signal)): # Iterate over the rest of the signals
                sigmat[:,signum] = signal['y'+str(signum)] # Write the rest of the signals

        with open(filepath, 'w+', newline='') as csvfile: # Write the waveform to the provided filepath
            writer = csv.writer(csvfile, delimiter=' ')
            csvfile.write(str(len(signal)-1)+'\n')
            writer.writerows(sigmat)
        self.f_name = filepath

class ElmRes(BasePFObject):
    
    LOADED, NOTLOADED = ("LOADED", "NOTLOADED")
    
    def __init__(self, pfobject, load = True):
        super().__init__(pfobject)
        if load:
            self.Load()
        else: 
            self.Release()
        self.head = []
        
    @property
    def state(self):
        return (ElmRes.LOADED if self._state == ElmRes.LOADED else ElmRes.NOTLOADED) 

    @state.setter
    def state(self, state):
        if state == ElmRes.LOADED:
            self.Load()
        elif state == ElmRes.NOTLOADED:
            self.Release()
            
    def Load(self):
        self.pfobject.Load()
        self._state = ElmRes.LOADED
    
    def Release(self):
        self.pfobject.Release()
        self._state = ElmRes.NOTLOADED
    
    @property
    def n_rows(self):
        if self.state == ElmRes.NOTLOADED:
            raise Exception(f'{self.name} is not loaded!')
        return self.GetNumberOfRows()
    
    @property
    def n_cols(self):
        if self.state == ElmRes.NOTLOADED:
            raise Exception(f'{self.name} is not loaded!')
        return self.GetNumberOfColumns()    
        
    @property
    def variables(self):
        if self.state == ElmRes.NOTLOADED:
            raise Exception(f'{self.name} is not loaded!')
        vars = defaultdict(list)
        for j in range(self.n_cols):
            elm = self.GetObject(j)
            var = self.GetVariable(j)
            vars[f'{elm.loc_name}.{elm.GetClassName()}'].append(var)
        return vars
    
    @variables.setter
    def variables(self, outputs):
        """Define variables of the ElmRes object for recording.
        
        Note
        ----
        The existing definition is reset every time this function is
        called. The ElmRes file is uloaded if it was loaded before.
        
        Parameters
        ----------
        outputs : dict
            Dictionary of variables which one wants to record in ElmRes.
        """
        self.clear_variables()
        for elm_name, var_names in outputs.items():
            for element in pfhandle.GetCalcRelevantObjects(elm_name.split('\\')[-1]):
                full_name = element.GetFullName()
                split_name = full_name.split('\\')
                full_name_reduced = []
                for dir in split_name[:-1]:
                    full_name_reduced.append(dir.split('.')[0])
                full_name_reduced.append(split_name[-1]) 
                full_name_reduced = '\\'.join(full_name_reduced)
                if not ((elm_name in full_name) or (elm_name in full_name_reduced)):
                    continue  
                for variable in var_names:
                    self.AddVariable(element, variable) # CHANGE TO THE NEWER VERSION
                    self.head.append(elm_name+'\\'+variable)
        self.InitialiseWriting()

    @property
    def outputs(self):
        """Returns the pandas data frame containing the simulation
        results.
        
        Returns
        -------
        DataFrame
            Pandas DataFrame containing the simulation results. The
            columns are monitored variables' names and the index is the
            simulation time.
        """
        if self.state == ElmRes.NOTLOADED:
            self.state = ElmRes.LOADED  
        # Get variable names
        var_values = np.array([[self.pfobject.GetValue(i, j)[1] for j in range(-1, self.n_cols)] for i in range(self.n_rows)])
        var_names = []
        for j in range(self.n_cols):
            elm = self.GetObject(j)
            var = self.GetVariable(j)
            var_names.append(f'{elm.loc_name}.{elm.GetClassName()}\\{var}')
        # Create Pandas data frame  
        results = pd.DataFrame(data=var_values[:,1:], index=var_values[:,0], columns=var_names)
        # Release from memory since it might be processed in the loop (check!)
        self.state = ElmRes.NOTLOADED
        return results

    def clear_variables(self):
        """Unloads the ElmRes file, deletes of the variable definitions,
        and intializes writing.
        """
        if self.state == ElmRes.LOADED:
            self.state = self.NOTLOADED
        for intmon in self.GetContents():
            intmon.Delete()
        self.InitialiseWriting()
        self.head = []

    @staticmethod
    def save_results(result, filepath):
        """Saves the pandas DataFrame of results as a JSON file.
        
        Parameters
        ----------
        result : DataFrame
            A DataFrame with simulation results.
        output_file: str
            A filepath at which the results are saved.
        """
        with open(filepath, 'w') as jsonfile:
            json.dump(result.iloc[:-1].to_json(), jsonfile)
            
class ComRes(BasePFObject):
    
    def __init__(self, pfobject, load = True):
            super().__init__(pfobject)
            self.head = [] # Header of the file
            self.col_Sep = ';' # Column separator
            self.dec_Sep = '.' # Decimal separator
            self.iopt_exp = 6 # Export type (csv)
            self.iopt_csel = 1 # Export only user defined vars
            self.ciopt_head = 1 # Use parameter names for variables
            self.iopt_sep = 0 # Don't use system separators
    
    def define_outputs(self, outputs, elmres, filepath):
        self.f_name = filepath
        # Adding time as first column
        resultobj = [elmres.pfobject]
        elements = [elmres.pfobject]
        cvariable = ['b:tnow']
        self.head = []
        # Defining all other results
        for elm_name, var_names in outputs.items():
            for element in pfhandle.GetCalcRelevantObjects(elm_name.split('\\')[-1]):
                full_name = element.GetFullName()
                split_name = full_name.split('\\')
                full_name_reduced = []
                for dir in split_name[:-1]:
                    full_name_reduced.append(dir.split('.')[0])
                full_name_reduced.append(split_name[-1]) 
                full_name_reduced = '\\'.join(full_name_reduced)
                if not ((elm_name in full_name) or (elm_name in full_name_reduced)):
                    continue  
                for variable in var_names:
                    self.head.append(elm_name+'\\'+variable)
                    elements.append(element)
                    cvariable.append(variable)
                    resultobj.append(elmres.pfobject)
        self.variable = cvariable
        self.resultobj = resultobj
        self.element = elements
        
    def read_results(self):
        self.ExportFullRange()
        return pd.read_csv(self.f_name, sep=self.col_Sep, skiprows=[0,1], index_col=0, names = self.head)
        
           
def parse_name(name):
    """Parses a name of a PowerFactory element and returns 
    its name and class"""

    pattern = '^(\w+)\.([a-zA-z]+$)'
    if match := re.match(pattern, name):
        return match.groups()
    else:
        raise ValueError

def change_parameters(params):
    """Sets the parameters of PF elements.
    
    Parameters
    ----------
    params : dict
        A dictionary of parameters of the following form {'element name 1':{'parameter name 1':value, ...}, ... }
    """
    for elm_name, par_dict in params.items(): # Iterate over network elements
        for param_name, value in par_dict.items():
            setattr(pf.GetCalcRelevantObjects(elm_name)[0], param_name, value) # Set the parameters of that element

def get_parameters(params):
    """Fetches the parameters of a model's elements.
    
    Parameters
    ----------
    params : dict
        A dictionary of parameters of the following form {'element name 1':['parameter name 1', ...], ... }
    """
    values = {}
    for elm_name, par_list in params.items(): # Iterate over network elements
        element = pfhandle.GetCalcRelevantObjects(elm_name)[0] # Get the corresponding element
        values[elm_name] = {}
        for param_name in par_list:
            values[elm_name][param_name] = getattr(element, param_name) # Add the extracted element to the dictionary
    return values

obj_map = {'ElmNet' : Network, 
       'IntCase' : StudyCase,
       'ElmRes' : ElmRes,
       'ElmFile' : ElmFile,
       'ComRes' : ComRes}

def set_params(params):
        """Sets the parameters of a model's elements.
        
        Parameters
        ----------
        params : dict
            A dictionary of parameters of the following form {'element name 1':{'parameter name 1':value, ...}, ... }
        """
        for elm_name, par_dict in params.items(): # Iterate over network elements
            for param_name, value in par_dict.items():
                setattr(pfhandle.GetCalcRelevantObjects(elm_name)[0], param_name, value) # Set the parameters of that element

def GetActiveNetworks():
    summary = pfhandle.GetActiveStudyCase().GetContents('Summary Grid.ElmNet')[0]
    return tuple(ref.obj_id for ref in summary.GetChildren(1))    

def generate_object_dict(obj_list, obj_class):
    return {obj.loc_name : obj_map[obj_class](obj) 
                for obj in obj_list
                if obj.GetClassName() == obj_class}

def print_tree(root):
    for line in _walk(root):
        print(line)

def _walk(root, lvl_chars=0):
    if children := root.GetContents():
        yield ' ' * lvl_chars + '| ' + f'{root.loc_name}.{root.GetClassName()}'
        lvl_chars += len(root.loc_name)//2 
        for child in children:
            yield from _walk(child, lvl_chars)
    else:
        yield ' ' * lvl_chars + '| ' + f'{root.loc_name}.{root.GetClassName()}'

def start_powerfactory():
    global pfhandle
    pfhandle = powerfactory.GetApplicationExt()


if __name__ == '__main__':
    start_powerfactory()
