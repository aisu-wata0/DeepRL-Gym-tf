
# Develop

To start you need to create a conda enviroment, this will install the correct python and packages versions in an isolated directory.

You need to install Miniconda for this <https://docs.conda.io/en/latest/miniconda.html> (if you have Anaconda that's also ok).

1. To create the enviroment do in the shell:

   ```bash
   conda env create -f enviroment_DRL.yml
   ```

2. Activate the enviroment in a shell

   ```bash
   conda activate DRL
   ```

3. To update your enviroment if later more packages are added to this repository dependencies:

   ```bash
   conda env update --file enviroment_DRL.yml --prune
   ```

4. To add more packages yourself, install the packages with conda or pip and then:

   ```bash
   conda env export --no-builds > enviroment_DRL.yml
   ```

## VSCode

To use the enviroment in VSCode press [Ctrl+P] (open the command pallete) and search for Python: Select Interpreter, then select the enviroment with the same name of the repository.

If it doesnt' appear make sure you created the enviroment and then restarted/reloaded VSCode

Select the linter 'mypy' ([Ctrl+P] > Python: Select linter > mypy).

"Mypy is an optional static type checker for Python that aims to combine the benefits of dynamic (or "duck") typing and static typing."
