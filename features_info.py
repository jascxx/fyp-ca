COLUMN_NAMES = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
                 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty']

CATEGORICAL_FEATURES = {'protocol_type': ['tcp', 'udp', 'icmp'], 
        'service': ['ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
                   'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
                   'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
                   'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
                   'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
                   'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
                   'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
                   'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
                   'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
                   'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i',
                   'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest'], 
        'flag': ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3',
                   'S2', 'OTH']}

BINARY_FEATURES = ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login']

INTRINSIC_FEATURES = COLUMN_NAMES[0:9]
CONTENT_FEATURES = COLUMN_NAMES[9:22]
TIME_BASED_TRAFFIC_FEATURES = COLUMN_NAMES[22:31]
HOST_BASED_TRAFFIC_FEATURES = COLUMN_NAMES[31:41]

flatten = lambda ls: [f for l in ls for f in l]
PROBE_FUNCTIONAL_FEATURES = flatten([INTRINSIC_FEATURES,
                                     TIME_BASED_TRAFFIC_FEATURES,
                                     HOST_BASED_TRAFFIC_FEATURES])
DOS_FUNCTIONAL_FEATURES = flatten([INTRINSIC_FEATURES,
                                   TIME_BASED_TRAFFIC_FEATURES])
U2R_FUNCTIONAL_FEATURES = flatten([INTRINSIC_FEATURES,
                                   CONTENT_FEATURES])
R2L_FUNCTIONAL_FEATURES = flatten([INTRINSIC_FEATURES,
                                   CONTENT_FEATURES])

# As we want the column names of the features, we map the categorical features to its values
# since we use one-hot encoding.
def map_to_one_hot(fs):
    col_names = []
    for f in fs:
        if f in CATEGORICAL_FEATURES:
            for cf in CATEGORICAL_FEATURES[f]:
                col_names.append(cf)
        else:
            col_names.append(f)
    return col_names
PROBE_FUNCTIONAL_COLS = map_to_one_hot(PROBE_FUNCTIONAL_FEATURES)
DOS_FUNCTIONAL_COLS = map_to_one_hot(DOS_FUNCTIONAL_FEATURES)
U2R_FUNCTIONAL_COLS = map_to_one_hot(U2R_FUNCTIONAL_FEATURES)
R2L_FUNCTIONAL_COLS = map_to_one_hot(R2L_FUNCTIONAL_FEATURES)


ATTACKS = {'DoS': ['apache2', 'smurf', 'neptune', 'back', 'teardrop', 'pod',
                   'land', 'mailbomb', 'processtable', 'udpstorm'],
           'R2L': ['warezclient', 'guess_passwd', 'warezmaster', 'imap',
                   'ftp_write', 'named', 'multihop', 'phf', 'spy', 'sendmail',
                   'snmpgetattack', 'snmpguess', 'worm', 'xsnoop', 'xlock'],
           'U2R': ['buffer_overflow', 'httptunnel', 'rootkit', 'loadmodule',
                   'perl', 'xterm', 'ps', 'sqlattack'],
           'Probe': ['satan', 'saint', 'ipsweep', 'portsweep', 'nmap', 'mscan']}
