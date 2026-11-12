"""Settings used by anon console app

"""
from typing import List

import yaml

from anonapi.exceptions import AnonAPIException
from anonapi.objects import RemoteAnonServer


class AnonClientSettings:
    """Settings used by anonymization web API client """

    def __init__(self, servers, user_name, user_token, job_default_parameters=None,
                 validate_ssl=True):
        """
        Parameters
        ----------
        servers: List[RemoteAnonServer]
            all servers
        user_name: str
            user name
        user_token: str
            API token
        job_default_parameters: JobDefaultParameters, optional
            When creating jobs, use these settings by default
        validate_ssl: bool, optional
            If False, ignore ssl warnings and outdated certificates.
            Defaults to True

        """
        self.servers = servers
        self.user_name = user_name
        self.user_token = user_token
        if job_default_parameters:
            self.job_default_parameters = job_default_parameters
        else:
            self.job_default_parameters = []
        if servers:
            self.active_server = servers[0]
        else:
            self.active_server = None
        self.validate_ssl = validate_ssl

    def to_datamap(self):
        """Convert these settings to a dict that can be used by YAML

        """
        datamap = {
            "servers": {x.name: x.url for x in self.servers},
            "user_name": self.user_name,
            "user_token": self.user_token,
            "validate_ssl": self.validate_ssl,
            "job_default_parameters": self.job_default_parameters.to_dict(),
        }
        if self.active_server:
            datamap["active_server_name"] = self.active_server.name
        else:
            datamap["active_server_name"] = None
        return datamap

    def as_human_readable(self) -> str:
        return yaml.dump(self.to_datamap(), default_flow_style=False)

    def save_to_file(self, filename):
        """ Putting save to file method here in base class so I can write settings files generated from code

        """
        datamap = self.to_datamap()
        with open(filename, "w") as f:
            yaml.dump(datamap, f, default_flow_style=False)

    def save(self):
        """ Implementing save() here to fulfil settings object signature
        """
        pass  # can't really save anything


class DefaultAnonClientSettings(AnonClientSettings):
    """A settings object with some default values. For testing and for writing settings when none are available

    """

    def __init__(self):
        """Create default settings object:

        >>> servers = [RemoteAnonServer("test", "https://hostname_of_api")]
        >>> user_name='username'
        >>> user_token='12345abc'

        """
        super().__init__(
            servers=[RemoteAnonServer("testserver", "https://hostname_of_api")],
            user_name="username",
            user_token="token",
            job_default_parameters=JobDefaultParameters(
                project_name="",  # these need to be set by user later
                destination_path="",
            ),
        )


NOT_SET = 'NOT_SET12342t4'  # sorry

class DataMap:
    """Structure to hold output from a yaml load(). Raises error when you cannot get()
     an expected key Poor man's substitute for schema validation.

    """

    def __init__(self, datamap):
        self._datamap = datamap

    def get(self, key, default=NOT_SET):
        if key not in self._datamap.keys():
            if default != NOT_SET:
                return default
            else:
                msg = f"expected to find key '{key}'"
                raise AnonClientSettingsFromFileException(msg)
        return self._datamap[key]

    def __iter__(self):
        return self._datamap.__iter__()


class AnonClientSettingsFromFile(AnonClientSettings):
    """ Settings which are bound to a file. Can load and save from there.

    """

    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            datamap = yaml.safe_load(f)

        self.parse_datamap(DataMap(datamap))

    def __str__(self):
        return f"Settings at {self.filename}"

    def parse_datamap(self, datamap: DataMap):
        try:
            servers = datamap.get("servers")
            user_name = datamap.get("user_name")
            user_token = datamap.get("user_token")
            active_server_name = datamap.get("active_server_name")
            validate_ssl = datamap.get("validate_ssl", default=True)
        except AnonClientSettingsFromFileException as e:
            msg = f"Could not read all settings from {self.filename}: {e}"
            raise AnonClientSettingsException(msg)

        if "job_default_parameters" in datamap:
            create_job_defaults_parsed = JobDefaultParameters.from_dict(
                datamap.get("job_default_parameters")
            )
        else:  # if this is an old settings file 'job_default_parameters' will
            # not exist. Just insert a default
            create_job_defaults_parsed = (
                DefaultAnonClientSettings().job_default_parameters
            )

        servers_parsed = {}
        for name, url in servers.items():
            servers_parsed[name] = RemoteAnonServer(name=name, url=url)

        super().__init__(
            servers=list(servers_parsed.values()),
            user_name=user_name,
            user_token=user_token,
            job_default_parameters=create_job_defaults_parsed,
            validate_ssl=validate_ssl
        )
        # set active server
        if active_server_name is None:
            self.active_server = None
        else:
            try:
                self.active_server = servers_parsed[active_server_name]
            except KeyError:
                msg = (
                    f"Active server name '{active_server_name}' was not found in list of servers_parsed "
                    f"'{list(servers_parsed.keys())}'. I don't know what the active server is supposed to be"
                )
                raise AnonClientSettingsException(msg)

    def save(self):
        super().save_to_file(self.filename)


class JobDefaultParameters:
    """Parameters that generally remain the same when creating jobs
    """

    def __init__(self, project_name, destination_path):
        """

        Parameters
        ----------
        project_name: str
        destination_path: Path
        """
        self.project_name = project_name
        self.destination_path = destination_path

    def to_dict(self):
        return {
            "project_name": self.project_name,
            "destination_path": self.destination_path,
        }

    @classmethod
    def from_dict(cls, dict_in):
        """

        Parameters
        ----------
        dict_in: dict

        Raises
        ------
        KeyError
            When expected parameter is missing from dict

        Returns
        -------
        JobDefaultParameters
        """
        return cls(
            project_name=dict_in["project_name"],
            destination_path=dict_in["destination_path"],
        )


class AnonClientSettingsException(AnonAPIException):
    pass


class AnonClientSettingsFromFileException(AnonClientSettingsException):
    pass
