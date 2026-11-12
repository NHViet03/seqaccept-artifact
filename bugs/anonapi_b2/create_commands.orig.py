"""Click group and commands for the 'create' subcommand
"""
from typing import List, Optional

import click

from anonapi.batch import BatchFolder, JobBatch
from anonapi.context import AnonAPIContext
from anonapi.client import APIClientException
from anonapi.decorators import pass_anonapi_context, handle_anonapi_exceptions
from anonapi.exceptions import AnonAPIException
from anonapi.mapper import MappingFolder, MapperException
from anonapi.parameters import (SourceIdentifier, StudyInstanceUIDIdentifier,
                                Parameter, DestinationPath, PatientID, PatientName,
                                Project, Description, SourceIdentifierParameter,
                                PIMSKey, ParameterSet, RootSourcePath, is_unc_path,
                                get_legacy_idis_value)
from anonapi.responses import JobInfo
from anonapi.settings import JobDefaultParameters, AnonClientSettingsException
from click.exceptions import Abort, ClickException

from pathlib import PureWindowsPath

from anonapi.testresources import JobInfoFactory


class JobParameterSet(ParameterSet):
    """A collection of parameters that should create one job.

    Offers validation and mapping to job-creation function keywords
    """

    # keywords to use for each Parameter.
    PARAMETER_KEYWORDS = {
        DestinationPath: "destination_path",
        PatientID: "anon_id",
        PatientName: "anon_name",
        Project: "project_name",
        Description: "description",
        PIMSKey: "pims_keyfile_id",
    }

    # these types of parameters are never sent to a function directly. They
    # should be ignored when casting to kwargs
    NON_KEYWORD_PARAMETERS = [RootSourcePath]

    @classmethod
    def is_non_keyword(cls, parameter):
        """Is this parameter of a type that is never sent as a parameter directly?"""
        return any(isinstance(parameter, x) for x in cls.NON_KEYWORD_PARAMETERS)

    def get_source(self) -> Optional[SourceIdentifierParameter]:
        """Get the parameter indicating the source of the data"""
        return self.get_param_by_type(SourceIdentifierParameter)

    def as_kwargs(self):
        """Parameters as keyword arguments

        Raises
        ------
        ParameterMappingException
            If not all parameters can be mapped

        Returns
        -------
        Dict[str: Parameter]
            Job creation function parameter: Parameter dict
            This can be put into a creation function with **as_kwargs(params)

        """
        dict_out = {}

        # make all parameter paths absolute
        try:
            absolute_parameters = self.with_unc_paths()
        except NoAbsoluteRootPathException as e:
            raise ParameterMappingException(e)

        for parameter in absolute_parameters:
            if self.is_non_keyword(parameter):
                # This parameter should not be included in kwargs. Skip
                continue
            elif self.is_source_identifier(parameter):
                if self.is_pacs_type(parameter):
                    dict_out["source_instance_id"] = \
                        get_legacy_idis_value(parameter.value)
                elif self.is_path_type(parameter):
                    dict_out["source_path"] = str(parameter.value)
                else:
                    raise ParameterMappingException(
                        f"Unknown source parameter '{parameter}'"
                    )
            else:
                try:
                    dict_out[self.PARAMETER_KEYWORDS[type(parameter)]] = str(
                        parameter.value
                    )
                except KeyError:
                    raise ParameterMappingException(f"Unknown parameter '{parameter}'")

        return dict_out

    def validate(self):
        """Assert that this set can be used to create a job

        Raises
        ------
        JobSetValidationError
            If this set has problems or is missing required values
        """

        for required in [SourceIdentifierParameter, DestinationPath, Project]:
            if not self.get_param_by_type(required):
                raise JobSetValidationError(f"Missing required parameter {required}")

        try:
            self.with_unc_paths()
        except ParameterMappingException as e:
            raise JobSetValidationError(
                f"Error: {e}. Source and destination need to be absolute windows"
                f" paths."
            )

    def with_unc_paths(self):
        """A copy of this JobParameterSet where all paths are absolute UNC
        paths. No relative paths, no mapped drive letters

        Raises
        ------
        ParameterMappingException
            If there are relative paths that cannot be resolved or are not unc
        """
        # make sure that all relative paths can be resolved
        absolute_params = []
        for param in self.parameters:
            if hasattr(param, "path") and param.path:
                # there is a path, try to make absolute and check unc
                if not param.path.is_absolute():
                    param = param.as_absolute(self.get_absolute_root_path())
                if not is_unc_path(param.path):
                    raise ParameterMappingException(
                        f"{param} is not a unc path. It will not be clear where this "
                        f"path is outside the current computer"
                    )
            absolute_params.append(param)

        return absolute_params

    def get_absolute_root_path(self) -> PureWindowsPath:
        """From this set, get the root path

        Raises
        ------
        NoAbsoluteRootPathException
            If there are relative paths that cannot be resolved

        """
        root_path = self.get_param_by_type(RootSourcePath)
        if not root_path:
            raise NoAbsoluteRootPathException("No absolute root root_path defined")
        elif not root_path.path.is_absolute():
            raise NoAbsoluteRootPathException(
                f"Root root_path {root_path} is not absolute"
            )
        else:
            return root_path.value


class CreateCommandsContext(AnonAPIContext):
    """Passed to all methods in the create group. Contains some additional methods
    over general context instance

    """

    def __init__(self, context: AnonAPIContext):
        super(CreateCommandsContext, self).__init__(
            client_tool=context.client_tool,
            settings=context.settings,
            current_dir=context.current_dir,
        )

    def default_parameters(self) -> List[Parameter]:
        """Default parameters from settings
        """
        defaults: JobDefaultParameters = self.settings.job_default_parameters
        return [
            DestinationPath(defaults.destination_path),
            Project(defaults.project_name),
        ]

    def create_job_for_element(self, parameters: List[Parameter]):
        """Create a job for the given parameters

        Parameters
        ----------
        parameters: List[Parameter]
            The parameters to use

        Raises
        ------
        JobCreationException
            If creating jobs fails for any reason

        Returns
        -------
        int
            job id created
        """

        row = JobParameterSet(parameters)
        source = row.get_source()
        if not source:
            raise JobCreationException(
                "No source identifier found. I can't create a job without knowing"
                " where to get the data"
            )

        try:
            if row.is_pacs_type(source):
                response = self.client_tool.create_pacs_job(
                    server=self.get_active_server(), **row.as_kwargs()
                )
            elif row.is_path_type(source):
                response = self.client_tool.create_path_job(
                    server=self.get_active_server(), **row.as_kwargs()
                )
            else:
                raise JobCreationException(f"Unknown source '{source}'")

        except (APIClientException, AnonClientSettingsException) as e:
            raise JobCreationException(f"Error creating job for source {source}: {e}")

        return response.job_id

    def add_to_batch(self, created_job_ids):
        """Add job ids as batch in current dir. If batch does not exist, create"""
        batch_folder = BatchFolder(path=self.current_dir)
        if batch_folder.has_batch():
            batch: JobBatch = batch_folder.load()
        else:
            batch = JobBatch(job_ids=[], server=self.get_active_server())
        if batch.server.url != self.get_active_server().url:
            click.echo(
                "A batch exists in this folder, but for a different server. "
                "Not saving job ids in batch"
            )
        else:
            click.echo("Saving job ids in batch in current folder")
            batch.job_ids = sorted(
                list(set(batch.job_ids) | set(created_job_ids))
            )  # add only unique new ids
            batch_folder.save(batch)

    @handle_anonapi_exceptions
    def get_mapping(self):
        return MappingFolder(self.current_dir).get_mapping()


pass_create_commands_context = click.make_pass_decorator(CreateCommandsContext)


@click.group(name="create")
@click.pass_context
@pass_anonapi_context
def main(context: AnonAPIContext, ctx):
    """create jobs"""
    ctx.obj = CreateCommandsContext(context=context)


@click.command()
@pass_create_commands_context
@click.option(
    "--dry-run/--no-dry-run", default=False, help="Do not post to server, just print"
)
def from_mapping(context: CreateCommandsContext, dry_run):
    """Create jobs from mapping in current folder"""
    if dry_run:
        click.echo("** Dry run, nothing will be sent to server **")
    mapping = context.get_mapping()

    # add defaults to each row
    job_sets = [
        JobParameterSet(row, default_parameters=context.default_parameters())
        for row in mapping.rows()
    ]
    # validate each job set
    for job_set in job_sets:
        try:
            job_set.validate()
        except JobSetValidationError as e:
            raise ClickException(f"Error validating parameters: {e}")

    # inspect project name and destination to present the next question to the user
    project_names = set()
    destination_paths = set()
    for job_set in job_sets:
        project_names.add(job_set.get_param_by_type(Project).value)
        destination_paths.add(job_set.get_param_by_type(DestinationPath).value)

    question = (
        f"This will create {len(mapping)} jobs on {context.get_active_server().name},"
        f" for projects '{list(project_names)}', writing data to "
        f"'{[str(x) for x in destination_paths]}'. Are you sure?"
    )
    if not click.confirm(question):
        click.echo("Cancelled")
        return

    created_job_ids = []
    for job_set in job_sets:
        if dry_run:

            def mock_create(*args, **kwargs):
                click.echo("create was called with rows:")
                click.echo("\n".join(args))
                click.echo("\n".join(map(str, kwargs.items())))
                return JobInfoFactory(job_id=-1)  # a mocked response

            context.client_tool.create_path_job = mock_create
            context.client_tool.create_pacs_job = mock_create
        try:
            job_id = context.create_job_for_element(job_set.parameters)
            click.echo(f"Created job with id {job_id}")
            created_job_ids.append(job_id)
        except JobCreationException as e:
            click.echo(str(e))
            click.echo(
                "Error will probably keep occurring. Stopping further job creation."
            )
            break

    click.echo(f"created {len(created_job_ids)} jobs: {created_job_ids}")

    if created_job_ids:
        context.add_to_batch(created_job_ids)

    click.echo("Done")


@click.command()
@pass_create_commands_context
def set_defaults(context: CreateCommandsContext):
    """Set project name used when creating jobs"""
    job_default_parameters: JobDefaultParameters = context.settings.job_default_parameters
    click.echo(
        "Please set default rows current value shown in [brackets]. Pressing enter"
        " without input will keep current value"
    )
    try:
        project_name = click.prompt(
            f"Please enter default IDIS project name:",
            show_default=True,
            default=job_default_parameters.project_name,
        )

        destination_path = click.prompt(
            f"Please enter default job destination directory:",
            show_default=True,
            default=job_default_parameters.destination_path,
        )
    except Abort:
        click.echo("Cancelled")

    job_default_parameters.project_name = project_name
    job_default_parameters.destination_path = destination_path
    context.settings.save()
    click.echo("Saved")


@click.command()
@pass_create_commands_context
def show_defaults(context: CreateCommandsContext):
    """show project name used when creating jobs"""

    job_default_parameters: JobDefaultParameters = context.settings.job_default_parameters
    click.echo(f"default IDIS project name: {job_default_parameters.project_name}")
    click.echo(
        f"default job destination directory: "
        f"{job_default_parameters.destination_path}"
    )


for func in [from_mapping, set_defaults, show_defaults]:
    main.add_command(func)


class JobCreationException(APIClientException):
    pass


class ParameterMappingException(AnonAPIException):
    pass


class NoAbsoluteRootPathException(ParameterMappingException):
    pass


class JobSetValidationError(AnonAPIException):
    pass
