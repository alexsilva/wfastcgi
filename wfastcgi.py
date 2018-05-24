# Python Tools for Visual Studio
# Copyright(c) Microsoft Corporation
# All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the License); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
# 
# THIS CODE IS PROVIDED ON AN  *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY
# IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# 
# See the Apache Version 2.0 License for specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import, print_function, with_statement

import base64
import ctypes
import logging
import logging.handlers
import os
import re
import socket
import struct
import sys
import traceback
import warnings
from xml.dom import minidom
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from cStringIO import StringIO
    BytesIO = StringIO
except ImportError:
    from io import StringIO, BytesIO
try:
    from thread import start_new_thread
except ImportError:
    from _thread import start_new_thread

if sys.version_info[0] == 3:
    def to_str(value):
        return value.decode(sys.getfilesystemencoding())
else:
    def to_str(value):
        return value.encode(sys.getfilesystemencoding())

__author__ = "Microsoft Corporation <ptvshelp@microsoft.com>"
__version__ = "3.3.4"

# http://www.fastcgi.com/devkit/doc/fcgi-spec.html#S3

FCGI_VERSION_1 = 1
FCGI_HEADER_LEN = 8

FCGI_BEGIN_REQUEST = 1
FCGI_ABORT_REQUEST = 2
FCGI_END_REQUEST = 3
FCGI_PARAMS = 4
FCGI_STDIN = 5
FCGI_STDOUT = 6
FCGI_STDERR = 7
FCGI_DATA = 8
FCGI_GET_VALUES = 9
FCGI_GET_VALUES_RESULT = 10
FCGI_UNKNOWN_TYPE = 11
FCGI_MAXTYPE = FCGI_UNKNOWN_TYPE

FCGI_NULL_REQUEST_ID = 0

FCGI_KEEP_CONN = 1

FCGI_RESPONDER = 1
FCGI_AUTHORIZER = 2
FCGI_FILTER = 3

FCGI_REQUEST_COMPLETE = 0
FCGI_CANT_MPX_CONN = 1
FCGI_OVERLOADED = 2
FCGI_UNKNOWN_ROLE = 3

FCGI_MAX_CONNS = "FCGI_MAX_CONNS"
FCGI_MAX_REQS = "FCGI_MAX_REQS"
FCGI_MPXS_CONNS = "FCGI_MPXS_CONNS"


class Logger(object):
    """ Main logger """
    formatter = '%(asctime)s %(levelname)s %(message)s'

    environ_name = "LOAD_BALANCE_HASH"

    def __init__(self, name):
        self.name = name
        # Each machine generates a unique hash
        self.hash = self.generate_hash()
        self.__init(name)

    def generate_hash(self):
        """Generates a new secure name to use in a load balance environment"""
        # noinspection PyBroadException
        try:
            # Use the machine's ip
            _hash = socket.gethostbyname(socket.gethostname())
        except:
            _hash = socket.gethostname()
        _hash = base64.urlsafe_b64encode(_hash)
        # The generated code can be accessed through the
        # environment variable WFASTCGI_LOAD_BALANCE_HASH
        os.environ[self.name.upper() + "_" + self.environ_name] = _hash
        return _hash

    def _get_boolean(self, name):
        try:
            is_enable = bool(int(os.environ.get(name, 0)))
        except ValueError:
            is_enable = False
        return is_enable

    @property
    def load_balance_enable(self):
        return self._get_boolean("WSGI_LOAD_BALANCE_ENABLE")

    @property
    def log_rotate_enable(self):
        return self._get_boolean("WSGI_LOG_ROTATE")

    def make_safe_filepath(self, filepath):
        """Generates a new path using the server's unique hash"""
        if not self.load_balance_enable or filepath is None:
            return filepath

        filepath = os.path.abspath(os.path.normpath(filepath))

        dirpath = os.path.dirname(filepath)

        return os.path.join(dirpath, self.hash + "_" + os.path.basename(filepath))

    def __init(self, name):
        self.logger = logging.getLogger(name)
        filepath = self.make_safe_filepath(os.environ.get("WSGI_LOG"))
        if not filepath:
            self._add_handler(logging.StreamHandler(stream=sys.stdout))
        elif self.log_rotate_enable:
            from cloghandler import ConcurrentRotatingFileHandler
            self._add_handler(ConcurrentRotatingFileHandler(
                filepath,
                maxBytes=int(os.environ.get('WSGI_LOG_ROTATE_MAXBYTES', 1024 ** 5)),
                backupCount=int(os.environ.get('WSGI_LOG_ROTATE_BACKUP_COUNT', 2)),
                lock_filename=self.make_safe_filepath(os.environ.get("WSGI_LOG_LOCK"))
            ))
        else:
            self._add_handler(logging.FileHandler(filepath))

    def _add_handler(self, handler):
        handler.setFormatter(logging.Formatter(os.environ.get('WSGI_LOG_FORMAT', self.formatter)))
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    def __call__(self, name=None):
        self.__init(name or self.name)

    def log(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    def debug(self, msg):
        self.log(msg, level=logging.DEBUG)


# wfastcgi
logger = Logger(os.path.basename(os.path.splitext(__file__)[0]))


class FastCgiRecord(object):
    """Represents a FastCgiRecord.  Encapulates the type, role, flags.  Holds
    onto the params which we will receive and update later."""

    def __init__(self, type, req_id, role, flags):
        self.type = type
        self.req_id = req_id
        self.role = role
        self.flags = flags
        self.params = {}

    def __repr__(self):
        return '<FastCgiRecord(%d, %d, %d, %d)>' % (self.type,
                                                    self.req_id,
                                                    self.role,
                                                    self.flags)


# typedef struct {
#   unsigned char version;
#   unsigned char type;
#   unsigned char requestIdB1;
#   unsigned char requestIdB0;
#   unsigned char contentLengthB1;
#   unsigned char contentLengthB0;
#   unsigned char paddingLength;
#   unsigned char reserved;
#   unsigned char contentData[contentLength];
#   unsigned char paddingData[paddingLength];
# } FCGI_Record;

class _ExitException(Exception):
    pass


if sys.version_info[0] >= 3:
    # indexing into byte strings gives us an int, so
    # ord is unnecessary on Python 3
    def ord(x):
        return x


    def chr(x):
        return bytes((x,))


    def wsgi_decode(x):
        return x.decode('iso-8859-1')


    def wsgi_encode(x):
        return x.encode('iso-8859-1')


    def fs_encode(x):
        return x


    def exception_with_traceback(exc_value, exc_tb):
        return exc_value.with_traceback(exc_tb)


    zero_bytes = bytes
else:
    # Replace the builtin open with one that supports an encoding parameter
    from codecs import open


    def wsgi_decode(x):
        return x


    def wsgi_encode(x):
        return x


    def fs_encode(x):
        return x if isinstance(x, str) else x.encode(sys.getfilesystemencoding())


    def exception_with_traceback(exc_value, exc_tb):
        # x.with_traceback() is not supported on 2.x
        return exc_value


    bytes = str


    def zero_bytes(length):
        return '\x00' * length


def read_fastcgi_record(stream):
    """reads the main fast cgi record"""
    try:
        data = stream.read(8)  # read record
    except IOError:
        data = ''  # pipe closed

    if not data:
        # no more data, our other process must have died...
        raise _ExitException()

    fcgi_ver, reqtype, req_id, content_size, padding_len, _ = struct.unpack('>BBHHBB', data)

    content = stream.read(content_size)  # read content
    stream.read(padding_len)

    if fcgi_ver != FCGI_VERSION_1:
        raise Exception('Unknown fastcgi version %s' % fcgi_ver)

    processor = REQUEST_PROCESSORS.get(reqtype)
    if processor is not None:
        return processor(stream, req_id, content)

    # unknown type requested, send response
    log('Unknown request type %s' % reqtype)
    send_response(stream, req_id, FCGI_UNKNOWN_TYPE, chr(reqtype) + zero_bytes(7))
    return None


def read_fastcgi_begin_request(stream, req_id, content):
    """reads the begin request body and updates our _REQUESTS table to include
    the new request"""
    #    typedef struct {
    #        unsigned char roleB1;
    #        unsigned char roleB0;
    #        unsigned char flags;
    #        unsigned char reserved[5];
    #    } FCGI_BeginRequestBody;

    # TODO: Ignore request if it exists
    res = FastCgiRecord(
        FCGI_BEGIN_REQUEST,
        req_id,
        (ord(content[0]) << 8) | ord(content[1]),  # role
        ord(content[2]),  # flags
    )
    _REQUESTS[req_id] = res


def read_encoded_int(content, offset):
    i = struct.unpack_from('>B', content, offset)[0]

    if i < 0x80:
        return offset + 1, i

    return offset + 4, struct.unpack_from('>I', content, offset)[0] & ~0x80000000


def read_fastcgi_keyvalue_pairs(content, offset):
    """Reads a FastCGI key/value pair stream"""

    offset, name_len = read_encoded_int(content, offset)
    offset, value_len = read_encoded_int(content, offset)

    name = content[offset:(offset + name_len)]
    offset += name_len

    value = content[offset:(offset + value_len)]
    offset += value_len

    return offset, name, value


def get_encoded_int(i):
    """Writes the length of a single name for a key or value in a key/value
    stream"""
    if i <= 0x7f:
        return struct.pack('>B', i)
    elif i < 0x80000000:
        return struct.pack('>I', i | 0x80000000)
    else:
        raise ValueError('cannot encode value %s (%x) because it is too large' % (i, i))


def write_fastcgi_keyvalue_pairs(pairs):
    """Creates a FastCGI key/value stream and returns it as a byte string"""
    parts = []
    for raw_key, raw_value in pairs.items():
        key = wsgi_encode(raw_key)
        value = wsgi_encode(raw_value)

        parts.append(get_encoded_int(len(key)))
        parts.append(get_encoded_int(len(value)))
        parts.append(key)
        parts.append(value)

    return bytes().join(parts)


# Keys in this set will be stored in the record without modification but with a
# 'wsgi.' prefix. The original key will have the decoded version.
# (Following mod_wsgi from http://wsgi.readthedocs.org/en/latest/python3.html)
RAW_VALUE_NAMES = {
    'SCRIPT_NAME': 'wsgi.script_name',
    'PATH_INFO': 'wsgi.path_info',
    'QUERY_STRING': 'wsgi.query_string',
    'HTTP_X_ORIGINAL_URL': 'wfastcgi.http_x_original_url',
}


def read_fastcgi_params(stream, req_id, content):
    if not content:
        return None

    offset = 0
    res = _REQUESTS[req_id].params
    while offset < len(content):
        offset, name, value = read_fastcgi_keyvalue_pairs(content, offset)
        name = wsgi_decode(name)
        raw_name = RAW_VALUE_NAMES.get(name)
        if raw_name:
            res[raw_name] = value
        res[name] = wsgi_decode(value)


def read_fastcgi_input(stream, req_id, content):
    """reads FastCGI std-in and stores it in wsgi.input passed in the
    wsgi environment array"""
    res = _REQUESTS[req_id].params
    if 'wsgi.input' not in res:
        res['wsgi.input'] = content
    else:
        res['wsgi.input'] += content

    if not content:
        # we've hit the end of the input stream, time to process input...
        return _REQUESTS[req_id]


def read_fastcgi_data(stream, req_id, content):
    """reads FastCGI data stream and publishes it as wsgi.data"""
    res = _REQUESTS[req_id].params
    if 'wsgi.data' not in res:
        res['wsgi.data'] = content
    else:
        res['wsgi.data'] += content


def read_fastcgi_abort_request(stream, req_id, content):
    """reads the wsgi abort request, which we ignore, we'll send the
    finish execution request anyway..."""
    pass


def read_fastcgi_get_values(stream, req_id, content):
    """reads the fastcgi request to get parameter values, and immediately 
    responds"""
    offset = 0
    request = {}
    while offset < len(content):
        offset, name, value = read_fastcgi_keyvalue_pairs(content, offset)
        request[name] = value

    response = {}
    if FCGI_MAX_CONNS in request:
        response[FCGI_MAX_CONNS] = '1'

    if FCGI_MAX_REQS in request:
        response[FCGI_MAX_REQS] = '1'

    if FCGI_MPXS_CONNS in request:
        response[FCGI_MPXS_CONNS] = '0'

    send_response(
        stream,
        req_id,
        FCGI_GET_VALUES_RESULT,
        write_fastcgi_keyvalue_pairs(response)
    )


# Our request processors for different FastCGI protocol requests. Only those
# requests that we receive are defined here.
REQUEST_PROCESSORS = {
    FCGI_BEGIN_REQUEST: read_fastcgi_begin_request,
    FCGI_ABORT_REQUEST: read_fastcgi_abort_request,
    FCGI_PARAMS: read_fastcgi_params,
    FCGI_STDIN: read_fastcgi_input,
    FCGI_DATA: read_fastcgi_data,
    FCGI_GET_VALUES: read_fastcgi_get_values
}

APPINSIGHT_CLIENT = None


def log(txt):
    """Logs messages to a log file if WSGI_LOG env var is defined."""
    if APPINSIGHT_CLIENT:
        try:
            APPINSIGHT_CLIENT.track_event(txt)
        except:
            pass

    logger.log(txt)


def maybe_log(txt):
    """Logs messages to a log file if WSGI_LOG env var is defined, and does not
    raise exceptions if logging fails."""
    try:
        log(txt)
    except:
        pass


def send_response(stream, req_id, resp_type, content, streaming=True):
    """sends a response w/ the given id, type, and content to the server.
    If the content is streaming then an empty record is sent at the end to 
    terminate the stream"""
    if not isinstance(content, bytes):
        raise TypeError("content must be encoded before sending: %r" % content)

    offset = 0
    while True:
        len_remaining = max(min(len(content) - offset, 0xFFFF), 0)

        data = struct.pack(
            '>BBHHBB',
            FCGI_VERSION_1,  # version
            resp_type,  # type
            req_id,  # requestIdB1:B0
            len_remaining,  # contentLengthB1:B0
            0,  # paddingLength
            0,  # reserved
        ) + content[offset:(offset + len_remaining)]

        offset += len_remaining

        os.write(stream.fileno(), data)
        if len_remaining == 0 or not streaming:
            break
    stream.flush()


def get_environment(dir):
    web_config = os.path.join(dir, 'Web.config')
    if not os.path.exists(web_config):
        return {}

    d = {}
    doc = minidom.parse(web_config)
    config = doc.getElementsByTagName('configuration')
    for configSection in config:
        appSettings = configSection.getElementsByTagName('appSettings')
        for appSettingsSection in appSettings:
            values = appSettingsSection.getElementsByTagName('add')
            for curAdd in values:
                key = curAdd.getAttribute('key')
                value = curAdd.getAttribute('value')
                if key and value is not None:
                    d[key.strip()] = value
    return d


ReadDirectoryChangesW = ctypes.windll.kernel32.ReadDirectoryChangesW
ReadDirectoryChangesW.restype = ctypes.c_uint32
ReadDirectoryChangesW.argtypes = [
    ctypes.c_void_p,  # HANDLE hDirectory
    ctypes.c_void_p,  # LPVOID lpBuffer
    ctypes.c_uint32,  # DWORD nBufferLength
    ctypes.c_uint32,  # BOOL bWatchSubtree
    ctypes.c_uint32,  # DWORD dwNotifyFilter
    ctypes.POINTER(ctypes.c_uint32),  # LPDWORD lpBytesReturned
    ctypes.c_void_p,  # LPOVERLAPPED lpOverlapped
    ctypes.c_void_p  # LPOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine
]
try:
    from _winapi import (CreateFile, CloseHandle, GetLastError, ExitProcess,
                         WaitForSingleObject, INFINITE, OPEN_EXISTING)
except ImportError:
    CreateFile = ctypes.windll.kernel32.CreateFileW
    CreateFile.restype = ctypes.c_void_p
    CreateFile.argtypes = [
        ctypes.c_wchar_p,  # lpFilename
        ctypes.c_uint32,  # dwDesiredAccess
        ctypes.c_uint32,  # dwShareMode
        ctypes.c_void_p,  # LPSECURITY_ATTRIBUTES,
        ctypes.c_uint32,  # dwCreationDisposition,
        ctypes.c_uint32,  # dwFlagsAndAttributes,
        ctypes.c_void_p  # hTemplateFile
    ]

    CloseHandle = ctypes.windll.kernel32.CloseHandle
    CloseHandle.argtypes = [ctypes.c_void_p]

    GetLastError = ctypes.windll.kernel32.GetLastError
    GetLastError.restype = ctypes.c_uint32

    ExitProcess = ctypes.windll.kernel32.ExitProcess
    ExitProcess.restype = ctypes.c_void_p
    ExitProcess.argtypes = [ctypes.c_uint32]

    WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
    WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    WaitForSingleObject.restype = ctypes.c_uint32

    OPEN_EXISTING = 3
    INFINITE = -1

FILE_LIST_DIRECTORY = 1
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
MAX_PATH = 260
FILE_NOTIFY_CHANGE_LAST_WRITE = 0x10
ERROR_NOTIFY_ENUM_DIR = 1022
INVALID_HANDLE_VALUE = 0xFFFFFFFF


class FILE_NOTIFY_INFORMATION(ctypes.Structure):
    _fields_ = [('NextEntryOffset', ctypes.c_uint32),
                ('Action', ctypes.c_uint32),
                ('FileNameLength', ctypes.c_uint32),
                ('Filename', ctypes.c_wchar)]


_ON_EXIT_TASKS = None


def run_exit_tasks():
    global _ON_EXIT_TASKS
    maybe_log("Running on_exit tasks")
    while _ON_EXIT_TASKS:
        tasks, _ON_EXIT_TASKS = _ON_EXIT_TASKS, []
        for t in tasks:
            try:
                t()
            except Exception:
                maybe_log("Error in exit task: " + traceback.format_exc())


def on_exit(task):
    global _ON_EXIT_TASKS
    if _ON_EXIT_TASKS is None:
        _ON_EXIT_TASKS = tasks = []
        try:
            evt = int(os.getenv('_FCGI_SHUTDOWN_EVENT_'))
        except (TypeError, ValueError):
            maybe_log("Could not wait on event %s" % os.getenv('_FCGI_SHUTDOWN_EVENT_'))
        else:
            def _wait_for_exit():
                WaitForSingleObject(evt, INFINITE)
                run_exit_tasks()
                ExitProcess(0)

            start_new_thread(_wait_for_exit, ())
    _ON_EXIT_TASKS.append(task)


class FileSystemEventHandlerImpl(FileSystemEventHandler):
    """Class that receives events when files are changed"""

    def __init__(self, restart_regex):
        self.restart_regex = restart_regex
        self.restart = re.compile(restart_regex)

    def on_modified(self, event):
        filename = event.src_path
        if self.restart.match(filename):
            log('wfastcgi.py exiting because %s has changed, matching %s' % (filename, self.restart_regex))
            # we call ExitProcess directly to quickly shutdown the whole process
            # because sys.exit(0) won't have an effect on the main thread.
            run_exit_tasks()
            ExitProcess(0)


def start_file_watcher(path, restart_regex):
    paths = filter(os.path.exists, path.split("|"))
    if restart_regex is None:
        restart_regex = ".*((\\.py)|(\\.config))$"
    elif not restart_regex or not paths:
        # restart regex or paths set to empty string, no restart behavior
        return

    log('wfastcgi.py will restart when files in %s are changed: %s' % (path, restart_regex))

    observer = Observer()
    event_handler = FileSystemEventHandlerImpl(restart_regex)
    try:
        recursive = bool(int(os.environ.get('WSGI_WATCHER_RECURSIVE')))
    except ValueError:
        recursive = False
    for path in paths:
        observer.schedule(event_handler, path, recursive=recursive)
    observer.start()


def get_wsgi_handler(handler_name):
    if not handler_name:
        raise Exception('WSGI_HANDLER env var must be set')

    if not isinstance(handler_name, str):
        handler_name = to_str(handler_name)

    module_name, _, callable_name = handler_name.rpartition('.')
    should_call = callable_name.endswith('()')
    callable_name = callable_name[:-2] if should_call else callable_name
    name_list = [(callable_name, should_call)]
    handler = None
    last_tb = ''

    while module_name:
        try:
            handler = __import__(module_name, fromlist=[name_list[0][0]])
            last_tb = ''
            for name, should_call in name_list:
                handler = getattr(handler, name)
                if should_call:
                    handler = handler()
            break
        except ImportError:
            module_name, _, callable_name = module_name.rpartition('.')
            should_call = callable_name.endswith('()')
            callable_name = callable_name[:-2] if should_call else callable_name
            name_list.insert(0, (callable_name, should_call))
            handler = None
            last_tb = ': ' + traceback.format_exc()

    if handler is None:
        raise ValueError('"%s" could not be imported%s' % (handler_name, last_tb))

    return handler


def _set_script_name(env, record):
    """configures the app as a site prefix (/LM/W3SVC/2/ROOT/app)"""
    if 'APPL_MD_PATH' in record.params and record.params['APPL_MD_PATH']:
        match = re.match("/LM/W\d+SVC/\d+/ROOT(.*)", record.params['APPL_MD_PATH'], re.I)
        path = match.group(1)
        if not os.environ.get('SCRIPT_NAME', '').startswith(path):
            os.environ['SCRIPT_NAME'] = env.get('SCRIPT_NAME', path)
    return env


def _expand_python_path(env):
    for path in (v for k, v in env.items() if k.lower() == 'pythonpath'):
        # Expand environment variables manually.
        expanded_path = re.sub('%(\\w+?)%',
                               lambda m: os.getenv(m.group(1), ''),
                               path)
        sys.path.extend(fs_encode(p) for p in expanded_path.split(';') if p)


def read_environment_vars(physical_path):
    """Reads the configuration keys given in the web.config script"""
    env = get_environment(physical_path)
    os.environ.update(env)
    _expand_python_path(env)
    return env


def read_wsgi_handler():
    global APPINSIGHT_CLIENT
    handler = get_wsgi_handler(os.getenv("WSGI_HANDLER"))
    instr_key = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")
    if instr_key:
        try:
            # Attempt the import after updating sys.path - sites must
            # include applicationinsights themselves.
            from applicationinsights.requests import WSGIApplication
        except ImportError:
            maybe_log("Failed to import applicationinsights: " + traceback.format_exc())
        else:
            handler = WSGIApplication(instr_key, handler)
            APPINSIGHT_CLIENT = handler.client
            # Ensure we will flush any remaining events when we exit
            on_exit(handler.client.flush)

    return handler


class handle_response(object):
    """A context manager for handling the response. This will ensure that
    exceptions in the handler are correctly reported, and the FastCGI request is
    properly terminated.
    """

    def __init__(self, stream, record, get_output, get_errors):
        self.stream = stream
        self.record = record
        self._get_output = get_output
        self._get_errors = get_errors
        self.error_message = ''
        self.fatal_errors = False
        self.physical_path = ''
        self.header_bytes = None
        self.sent_headers = False

    def __enter__(self):
        record = self.record
        record.params['wsgi.input'] = BytesIO(record.params['wsgi.input'])
        record.params['wsgi.version'] = (1, 0)
        record.params['wsgi.url_scheme'] = 'https' if record.params.get('HTTPS', '').lower() == 'on' else 'http'
        record.params['wsgi.multiprocess'] = True
        record.params['wsgi.multithread'] = False
        record.params['wsgi.run_once'] = False

        self.physical_path = record.params.get('APPL_PHYSICAL_PATH', os.path.dirname(__file__))

        if 'HTTP_X_ORIGINAL_URL' in record.params:
            # We've been re-written for shared FastCGI hosting, so send the
            # original URL as PATH_INFO.
            record.params['PATH_INFO'] = record.params['HTTP_X_ORIGINAL_URL']
            record.params['wsgi.path_info'] = record.params['wfastcgi.http_x_original_url']

        # PATH_INFO is not supposed to include the query parameters, so remove them
        record.params['PATH_INFO'] = record.params['PATH_INFO'].partition('?')[0]
        record.params['wsgi.path_info'] = record.params['wsgi.path_info'].partition(wsgi_encode('?'))[0]

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Send any error message on FCGI_STDERR.
        if exc_type and exc_type is not _ExitException:
            error_msg = "%s:\n\n%s\n\nStdOut: %s\n\nStdErr: %s" % (
                self.error_message or 'Error occurred',
                ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                self._get_output(),
                self._get_errors(),
            )
            if not self.header_bytes or not self.sent_headers:
                self.header_bytes = wsgi_encode('Status: 500 Internal Server Error\r\n')
            self.send(FCGI_STDERR, wsgi_encode(error_msg))
            # Best effort at writing to the log. It's more important to
            # finish the response or the user will only see a generic 500
            # error.
            maybe_log(error_msg)
        else:
            # Prints the output of data to the log.
            for output in self._get_output().split("\n"):
                output = output.strip()
                if output:
                    maybe_log(output)

        # End the request. This has to run in both success and failure cases.
        self.send(FCGI_END_REQUEST, zero_bytes(8), streaming=False)

        # Remove the request from our global dict
        del _REQUESTS[self.record.req_id]

        # Suppress all exceptions unless requested
        return not self.fatal_errors

    @staticmethod
    def _decode_header(key, value):
        if not isinstance(key, str):
            key = wsgi_decode(key)
        if not isinstance(value, str):
            value = wsgi_decode(value)
        return key, value

    def start(self, status, headers, exc_info=None):
        """Starts sending the response. The response is ended when the context
        manager exits."""
        if exc_info:
            try:
                if self.sent_headers:
                    # We have to re-raise if we've already started sending data.
                    raise exception_with_traceback(exc_info[1], exc_info[2])
            finally:
                exc_info = None
        elif self.header_bytes:
            raise Exception('start_response has already been called')

        if not isinstance(status, str):
            status = wsgi_decode(status)
        header_text = 'Status: %s\r\n' % status
        if headers:
            header_text += ''.join('%s: %s\r\n' % handle_response._decode_header(*i) for i in headers)
        self.header_bytes = wsgi_encode(header_text + '\r\n')

        return lambda content: self.send(FCGI_STDOUT, content)

    def send(self, resp_type, content, streaming=True):
        '''Sends part of the response.'''
        if not self.sent_headers:
            if not self.header_bytes:
                raise Exception("start_response has not yet been called")

            self.sent_headers = True
            send_response(self.stream, self.record.req_id, FCGI_STDOUT, self.header_bytes)
            self.header_bytes = None

        return send_response(self.stream, self.record.req_id, resp_type, content, streaming)


_REQUESTS = {}


def _load_init_config(physical_path):
    # load env vars
    if os.path.isdir(physical_path):
        env = read_environment_vars(physical_path)
        # setup logs
        logger()
    else:
        env = None
    return env


def main(*args):
    # When the script was called with the wsgi main directory
    env = _load_init_config(args[0])

    initialized = False

    log('wfastcgi.py %s started' % __version__)
    log('Python version: %s' % sys.version)

    try:
        fcgi_stream = sys.stdin.detach() if sys.version_info[0] >= 3 else sys.stdin
        try:
            import msvcrt
            msvcrt.setmode(fcgi_stream.fileno(), os.O_BINARY)
        except ImportError:
            pass

        while True:
            record = read_fastcgi_record(fcgi_stream)
            if not record:
                continue

            errors = sys.stderr = sys.__stderr__ = record.params['wsgi.errors'] = StringIO()
            output = sys.stdout = sys.__stdout__ = StringIO()

            with handle_response(fcgi_stream, record, output.getvalue, errors.getvalue) as response:
                if not initialized:
                    log('wfastcgi.py %s initializing' % __version__)

                    os.chdir(response.physical_path)
                    sys.path[0] = '.'

                    if env is None:
                        env = _load_init_config(response.physical_path)

                    env = _set_script_name(env, record)

                    # Initialization errors should be treated as fatal.
                    response.fatal_errors = True
                    response.error_message = 'Error occurred while reading WSGI handler'
                    handler = read_wsgi_handler()

                    response.error_message = 'Error occurred starting file watcher'
                    start_file_watcher(env.get('WSGI_WATCHER_DIRS', response.physical_path),
                                       env.get('WSGI_RESTART_FILE_REGEX'))

                    # Enable debugging if possible. Default to local-only, but
                    # allow a web.config to override where we listen
                    ptvsd_secret = env.get('WSGI_PTVSD_SECRET')
                    if ptvsd_secret:
                        ptvsd_address = (env.get('WSGI_PTVSD_ADDRESS') or 'localhost:5678').split(':', 2)
                        try:
                            ptvsd_port = int(ptvsd_address[1])
                        except LookupError:
                            ptvsd_port = 5678
                        except ValueError:
                            log('"%s" is not a valid port number for debugging' % ptvsd_address[1])
                            ptvsd_port = 0

                        if ptvsd_address[0] and ptvsd_port:
                            try:
                                import ptvsd
                            except ImportError:
                                log('unable to import ptvsd to enable debugging')
                            else:
                                addr = ptvsd_address[0], ptvsd_port
                                ptvsd.enable_attach(secret=ptvsd_secret, address=addr)
                                log('debugging enabled on %s:%s' % addr)

                    response.error_message = ''
                    response.fatal_errors = False

                    log('wfastcgi.py %s initialized' % __version__)
                    initialized = True

                os.environ.update(env)

                # SCRIPT_NAME + PATH_INFO is supposed to be the full path
                # (http://www.python.org/dev/peps/pep-0333/) but by default
                # (http://msdn.microsoft.com/en-us/library/ms525840(v=vs.90).aspx)
                # IIS is sending us the full URL in PATH_INFO, so we need to
                # clear the script name here
                if 'AllowPathInfoForScriptMappings' not in os.environ:
                    record.params['SCRIPT_NAME'] = ''
                    record.params['wsgi.script_name'] = wsgi_encode('')

                # correct SCRIPT_NAME and PATH_INFO if we are told what our SCRIPT_NAME should be
                if 'SCRIPT_NAME' in os.environ and record.params['PATH_INFO'].lower().startswith(
                        os.environ['SCRIPT_NAME'].lower()):
                    record.params['SCRIPT_NAME'] = os.environ['SCRIPT_NAME']
                    record.params['PATH_INFO'] = record.params['PATH_INFO'][len(record.params['SCRIPT_NAME']):]
                    record.params['wsgi.script_name'] = wsgi_encode(record.params['SCRIPT_NAME'])
                    record.params['wsgi.path_info'] = wsgi_encode(record.params['PATH_INFO'])

                # Send each part of the response to FCGI_STDOUT.
                # Exceptions raised in the handler will be logged by the context
                # manager and we will then wait for the next record.

                result = handler(record.params, response.start)
                try:
                    for part in result:
                        if part:
                            response.send(FCGI_STDOUT, part)
                finally:
                    if hasattr(result, 'close'):
                        result.close()
    except _ExitException:
        pass
    except Exception:
        maybe_log('Unhandled exception in wfastcgi.py: ' + traceback.format_exc())
    except BaseException:
        maybe_log('Unhandled exception in wfastcgi.py: ' + traceback.format_exc())
        raise
    finally:
        run_exit_tasks()
        maybe_log('wfastcgi.py %s closed' % __version__)


def get_iis_version():
    """Returns the version of IIS installed on the system (x.x)"""
    try:
        import _winreg as winreg
    except ImportError:
        import winreg
    msg = ">> Unable to determine iis version"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\InetStp') as key:
            try:
                major = winreg.QueryValueEx(key, "MajorVersion")[0]
                minor = winreg.QueryValueEx(key, "MinorVersion")[0]
                version = float("{!s}.{!s}".format(major, minor))
            except (IndexError, ValueError):
                warnings.warn(msg)
                version = 0.0
    except WindowsError:
        warnings.warn(msg)
        version = 0.0
    return version


def _run_appcmd(args):
    from subprocess import check_call, CalledProcessError

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        appcmd = sys.argv[1:]
    else:
        appcmd = [os.path.join(os.getenv('SystemRoot'), 'system32', 'inetsrv', 'appcmd.exe')]

    if not os.path.isfile(appcmd[0]):
        print('IIS configuration tool appcmd.exe was not found at', appcmd, file=sys.stderr)
        return -1

    args = appcmd + args
    try:
        return check_call(args)
    except CalledProcessError as ex:
        print('''An error occurred running the command:

%r

Ensure your user has sufficient privileges and try again.''' % args, file=sys.stderr)
        return ex.returncode


def get_filepath():
    """Returns the absolute path of the script with the 'py' extension"""
    # Avoid using the script with the pyc extension
    return os.path.splitext(__file__)[0] + '.py'


def quoted_path(path):
    return '"' + path + '"' if ' ' in path else path


IIS_VERSION = 7.5


def get_app_cmd(root_dir, add=False, **kwargs):
    cmd = "/{operation}[fullPath='{executable}', arguments='{script}{root}', {extra}]"
    # defaults
    if get_iis_version() >= IIS_VERSION:
        # https://www.iis.net/configreference/system.webserver/fastcgi
        kwargs.setdefault('signalBeforeTerminateSeconds', '30')
    quoted_root = quoted_path(root_dir)
    options = {
        'operation': "+" if add else "-",
        'executable': quoted_path(sys.executable),
        'script': quoted_path(get_filepath()),
        'root': ' ' + quoted_root if quoted_root else quoted_root,
        'extra': ', '.join(["{}='{}'".format(*arg) for arg in kwargs.items()])
    }
    return cmd.format(**options)


def enable(root_dir, **kwargs):
    res = _run_appcmd([
        "set", "config", "/section:system.webServer/fastCGI",
        get_app_cmd(root_dir, add=True, **kwargs)
    ])
    if res == 0:
        print('"%s|%s" can now be used as a FastCGI script processor' % (sys.executable, get_filepath()))
    return res


def disable(root_dir, **kwargs):
    res = _run_appcmd([
        "set", "config", "/section:system.webServer/fastCGI",
        get_app_cmd(root_dir, add=False, **kwargs)
    ])
    if res == 0:
        print('"%s|%s" is no longer registered for use with FastCGI' % (sys.executable, get_filepath()))
    return res


if __name__ == '__main__':
    main(*sys.argv[1:])
