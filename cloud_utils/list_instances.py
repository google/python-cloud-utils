# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

"""list instances from AWS, and GCP."""
import sys

from argparse import ArgumentParser
import boto3
import botocore
from botocore.exceptions import ClientError
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import parse as parse_date
from functools import partial
import kubernetes.client.rest
import kubernetes.config.config_exception
from pytz import UTC
import re
from texttable import Texttable

try:
    from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
    import google.auth.exceptions  # pylint: disable=g-import-not-at-top
    from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
    from googleapiclient import errors  # pylint: disable=g-import-not-at-top
except ImportError:
    print("Warning, google-api-python-client or google-auth not installed, can't connect to GCP instances.")

try:
    from kubernetes import client, config  # pylint: disable=g-import-not-at-top,g-multiple-import
except ImportError:
    print("Warning, kubernetes not installed, not getting pods")

AUTOSCALING_GROUP_TAG = u'aws:autoscaling:groupName'
AWS_REGION_LIST = ['us-east-1', 'eu-west-1', 'us-west-2',
                   'sa-east-1', 'ap-southeast-1', 'eu-central-1']
GCP_REGION_LIST = ['us-east1', 'asia-east1', 'asia-northeast1',
                   'asia-southeast1', 'europe-west1', 'us-central1', 'us-west1']
INSTANCE_FILEDS = ['cloud', 'created', 'name', 'region', 'id', 'state', 'ip_address', 'type', 'autoscaling_group',
                   'iam_or_service_account', 'public_dns_name', 'private_ip_address', 'project', 'zone', 'security_groups',
                   'tags', 'vpc_id', 'reservation_type']
DEFAULT_SHOWN_FIELDS = ['project', 'name', 'zone', 'id', 'state', 'ip_address', 'type', 'created',
                        'autoscaling_group', 'iam_or_service_account', 'private_ip_address']
GCP_INSTANCE_TYPE_DICT = {'standard': 'std', 'highmem': 'mem', 'n1-': '', 'highcpu': 'cpu', 'custom': 'ctm'}
ALIGN_OPTIONS = ['l', 'c', 'r']
Instance = namedtuple('Instance', INSTANCE_FILEDS)
SUPPORTED_CLOUDS = ('gcp', 'aws', 'k8s')


def pretify_field(field):
    return ' '.join([word[0].upper() + word[1:] for word in field.split('_')])


def aws_get_instances_by_name(region, name, raw=True):
    """Get all instances in a region matching a name (with wildcards)."""
    return _aws_get_instance_by_tag(region, name, 'tag:Name', raw)


def aws_get_instances_by_autoscaling_group_in_region(region, name, raw=True):
    return _aws_get_instance_by_tag(region, name, 'tag:aws:autoscaling:groupName', raw)


def datetime_to_str(date):
    return date.astimezone(UTC).isoformat().split('+')[0].split('.')[0].replace('T', ' ')


def _aws_instance_from_dict(region, instance, raw):
    tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
    return Instance(cloud='aws',  # pylint: disable=expression-not-assigned
                    zone=instance['Placement']['AvailabilityZone'],
                    region=region,
                    id=instance['InstanceId'],
                    name=tags.get('Name', 'Empty'),
                    state=instance['State']['Name'],
                    type=instance['InstanceType'] if raw else instance['InstanceType'].replace('xlarge', 'xl'),
                    autoscaling_group=tags.get(AUTOSCALING_GROUP_TAG, None),
                    public_dns_name=instance['PublicDnsName'],
                    ip_address=instance.get('PublicIpAddress'),
                    iam_or_service_account=instance['IamInstanceProfile'][
                        'Arn'].split('/')[1] if instance.get('IamInstanceProfile') else None,
                    private_ip_address=instance.get('PrivateIpAddress'),
                    project='aws',
                    security_groups=[group['GroupName'] for group in instance['SecurityGroups']],
                    tags=tags,
                    created=datetime_to_str(instance['LaunchTime']),
                    reservation_type=instance.get('InstanceLifecycle', 'on_demand'),
                    vpc_id=instance.get('VpcId'))


def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    client = boto3.session.Session().client('ec2', region)
    matching_reservations = client.describe_instances(Filters=[{'Name': tag, 'Values': [name]}]).get('Reservations', [])
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances


def gcp_filter_projects_instances(projects, filters, raw=True, credentials=None):
    try:
        credentials = credentials or GoogleCredentials.get_application_default()
    except google.auth.exceptions.DefaultCredentialsError:
        return None

    if projects is None:
        resourcemanager = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
        projects = [project['projectId']
                    for project in resourcemanager.projects().list().execute().get('projects', [])]

    compute = discovery.build('compute', 'v1', credentials=credentials)
    batch = compute.new_batch_http_request()
    results = []
    replies = []
    for project in projects:
        for filter_to_apply in filters:
            batch.add(compute.instances().aggregatedList(project=project,
                                                         filter=filter_to_apply),
                      callback=partial(lambda request_id, result, exception, inside_filter, inside_project: replies.append((inside_project, inside_filter, result)),
                                       inside_filter=filter_to_apply, inside_project=project))
    batch.execute()

    results.extend([result for project, filter, result in replies])
    batch = compute.new_batch_http_request()
    next_batch_requests = [(project, filter, result)
                           for project, filter, result in replies if result and result.get('nextPageToken')]
    replies = []
    while next_batch_requests:
        [batch.add(compute.instances().aggregatedList(project=project,
                                                      filter=filter,
                                                      pageToken=result['nextPageToken']),
                   callback=partial(lambda request_id, result, exception, inside_filter, inside_project: replies.append((inside_project, inside_filter, result)),
                                    inside_filter=filter, inside_project=project))
         for project, filter, result in next_batch_requests]
        batch.execute()
        results.extend([result for project, filter, result in replies])
        batch = compute.new_batch_http_request()
        next_batch_requests = [(project, filter, result)
                               for project, filter, result in replies if result and result.get('nextPageToken')]
        replies = []

    instances = []
    [instances.extend(get_instace_object_from_gcp_list(result['id'].split('/', 2)[1],  # pylint: disable=expression-not-assigned
                                                       raw,
                                                       result.get('items', {})))
     for result in results if result is not None]
    return instances


def gcp_get_instances_by_label(project, name, raw=True, credentials=None):
    return gcp_filter_projects_instances(projects=[project],
                                         filters=['labels.name eq {name}'.format(name=name.replace('*', '.*'))],
                                         raw=raw,
                                         credentials=credentials)


def gcp_get_instances_by_name(project, name, raw=True, credentials=None):
    """Get instances in a GCP region matching name (with * wildcards)."""
    return gcp_filter_projects_instances(projects=[project],
                                         filters=['name eq {name}'.format(name=name.replace('*', '.*'))],
                                         raw=raw,
                                         credentials=credentials)


def get_instace_object_from_gcp_list(project, raw, region_to_instances):
    matching_intsances = []
    for region, region_instances in region_to_instances.items():
        for matching_instance in region_instances.get('instances', []):
            created_by = [item['value'] for item in matching_instance.get(
                'metadata', {}).get('items', []) if item['key'] == 'created-by']
            if created_by and 'instanceGroupManagers' in created_by[0]:
                if raw:
                    autoscaling_group = created_by[0]
                else:
                    autoscaling_group = created_by[0].split('/')[-1]
            else:
                autoscaling_group = None
            public_ip = None
            if raw:
                iam_or_service_account = ','.join([account['email']
                                                   for account in matching_instance.get('serviceAccounts', [])])
                instance_type = matching_instance['machineType'].split('/')[-1]
            else:
                iam_or_service_account = ','.join([account['email'].split('@')[0]
                                                   for account in matching_instance.get('serviceAccounts', [])])
                instance_type = matching_instance['machineType'].split('/')[-1]
                for raw_type, pretty_type in GCP_INSTANCE_TYPE_DICT.items():
                    instance_type = instance_type.replace(raw_type, pretty_type)
            for interface in matching_instance['networkInterfaces']:
                if 'accessConfigs' in interface and interface['accessConfigs'] and interface['accessConfigs'][0].get('natIP'):
                    public_ip = interface['accessConfigs'][0]['natIP']
            tags = {metadata['key']: metadata['value'] for metadata in matching_instance['metadata'].get('items', [])}
            tags.update(matching_instance.get('labels', {}))
            tags[None] = matching_instance['tags'].get('items', [])

            creation_datetime = datetime_to_str(parse_date(matching_instance['creationTimestamp']))
            matching_intsances.append(Instance(cloud='gcp',
                                               zone=region.replace('zones/', ''),
                                               region=region.replace('zones/', '')[:-2],
                                               id=matching_instance['name'],
                                               name=matching_instance['name'] if 'name' not in tags else tags['name'],
                                               state=matching_instance['status'].lower(),
                                               type=instance_type,
                                               autoscaling_group=autoscaling_group,
                                               public_dns_name=None,
                                               ip_address=public_ip,
                                               iam_or_service_account=iam_or_service_account,
                                               private_ip_address=matching_instance['networkInterfaces'][0].get(
                                                   'networkIP'),
                                               project=project,
                                               security_groups=[],
                                               tags=tags,
                                               created=creation_datetime,
                                               reservation_type='preemptible' if matching_instance.get(
                                                   'preemptible') else 'on-demand',
                                               vpc_id=None))
    return matching_intsances


def aws_get_instances_by_id(region, instance_id, raw=True):
    """Returns instances mathing an id."""
    client = boto3.session.Session().client('ec2', region)
    try:
        matching_reservations = client.describe_instances(InstanceIds=[instance_id]).get('Reservations', [])
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') != 'InvalidInstanceID.NotFound':
            raise
        return []
    instances = []
    [[instances.append(_aws_instance_from_dict(region, instance, raw))  # pylint: disable=expression-not-assigned
      for instance in reservation.get('Instances')] for reservation in matching_reservations if reservation]
    return instances


def get_instances_by_id(instance_id, regions=None, projects=None, raw=True, sort_by_order=('cloud', 'name'), clouds=SUPPORTED_CLOUDS):  # pylint: disable=unused-argument
    instance_id = instance_id.replace('*', '')
    with ThreadPoolExecutor(max_workers=len(AWS_REGION_LIST)) as executor:
        if 'aws' in clouds:
            results = executor.map(partial(aws_get_instances_by_id, instance_id=instance_id),
                                   AWS_REGION_LIST)
        else:
            results = []
    matching_instances = []
    for result in results:
        matching_instances.extend(result)
    if regions:
        matching_instances = [instance for instance in matching_instances if instance.region in regions]
    return matching_instances


def all_clouds_get_instances_by_name(name, projects, raw, credentials, clouds=SUPPORTED_CLOUDS):
    with ThreadPoolExecutor(max_workers=2 * len(AWS_REGION_LIST) + 2) as executor:
        if 'aws' in clouds:
            aws_name_results = executor.map(partial(aws_get_instances_by_name, name=name, raw=raw),
                                            AWS_REGION_LIST)
            aws_autoscaling_results = executor.map(partial(aws_get_instances_by_autoscaling_group_in_region, name=name, raw=raw),
                                                   AWS_REGION_LIST)
        if 'gcp' in clouds:
            gcp_job = executor.submit(gcp_filter_projects_instances,
                                      projects=projects,
                                      filters=['labels.name eq {name}'.format(name=name.replace('*', '.*')),
                                               'name eq {name}'.format(name=name.replace('*', '.*'))],
                                      raw=raw,
                                      credentials=credentials)
        if 'k8s' in clouds:
            kube_job = executor.submit(get_all_pods, name)

    matching_instances = []
    if 'gcp' in clouds:
        matching_instances = gcp_job.result()
    if 'aws' in clouds:
        for instances in [result for result in aws_name_results] + [result for result in aws_autoscaling_results]:
            # for instances in results:
            matching_instances.extend(instances)
    if 'k8s' in clouds:
        matching_instances.extend(kube_job.result())

    # Filter instances seen more than once (matching both autoscaling group and name)
    seen_instances = set()
    matching_instances = [instance
                          for instance in matching_instances if instance.id not in seen_instances and (
                              seen_instances.add(instance.id) or True)]
    return matching_instances


def get_instances_by_name(name, sort_by_order=('cloud', 'name'), projects=None, raw=True, regions=None, gcp_credentials=None, clouds=SUPPORTED_CLOUDS):
    """Get intsances from GCP and AWS by name."""
    matching_instances = all_clouds_get_instances_by_name(
        name, projects, raw, credentials=gcp_credentials, clouds=clouds)
    if regions:
        matching_instances = [instance for instance in matching_instances if instance.region in regions]
    matching_instances.sort(key=lambda instance: [getattr(instance, field) for field in sort_by_order])
    return matching_instances


def get_instances_by_id_or_name(identifier, *args, **kwargs):
    id_regex = re.compile(r'^\*?i?-?(?P<id>[0-9a-f]{17}|[0-9a-f]{8})\*?$')
    if id_regex.match(identifier):
        return get_instances_by_id(identifier, *args, **kwargs)
    return get_instances_by_name(identifier, *args, **kwargs)


def run_map(func, args, max_workers=None):
    def wrapper(arg):
        result = func(arg)
        return arg, result

    with ThreadPoolExecutor(max_workers or len(args)) as executor:
        results = executor.map(wrapper, args)

    return {result[0]: result[1] for result in results}


def get_pods_for_context(context, name):
    try:
        kube = client.CoreV1Api(config.new_client_from_config(context=context))
        pod_dicts = [pod.to_dict()
                     for pod in kube.list_pod_for_all_namespaces(watch=False,
                                                                 field_selector='metadata.namespace!=kube-system').items]
    except (kubernetes.client.rest.ApiException, kubernetes.config.config_exception.ConfigException):
        return []
    return [pod for pod in pod_dicts if re.match(name, pod['metadata']['name'])]


def get_all_pods(name):
    name = name.replace('*', '.*')
    try:
        all_contexts = [context['name'] for context in config.list_kube_config_contexts()[0]]
    except:  # pylint:disable=bare-except
        return []
    results = run_map(partial(get_pods_for_context, name=name), all_contexts, max_workers=2)
    instances = []
    [[instances.append(Instance(cloud='k8s',  # pylint: disable=expression-not-assigned
                                zone=context,
                                region=context,
                                id=pod['metadata']['name'],
                                name=pod['metadata']['name'],
                                state=pod['status']['phase'].lower(),
                                type='pod',
                                autoscaling_group=None,
                                public_dns_name=None,
                                ip_address=None,
                                iam_or_service_account=None,
                                private_ip_address=None,
                                project='k8s',
                                security_groups=None,
                                tags=None,
                                created=datetime_to_str(pod['status']['start_time']),
                                reservation_type='container',
                                vpc_id=None))
      for pod in pods]
     for context, pods in results.items()]
    return instances


def get_os_version(instance):
    """Get OS Version for instances."""
    if instance.cloud == 'aws':
        client = boto3.client('ec2', instance.region)
        image_id = client.describe_instances(InstanceIds=[instance.id])['Reservations'][0]['Instances'][0]['ImageId']
        return '16.04' if '16.04' in client.describe_images(ImageIds=[image_id])['Images'][0]['Name'] else '14.04'
    if instance.cloud == 'gcp':
        credentials = GoogleCredentials.get_application_default()
        compute = discovery.build('compute', 'v1', credentials=credentials)
        for disk in compute.instances().get(instance=instance.name,
                                            zone=instance.zone,
                                            project=instance.project).execute()['disks']:
            if not disk.get('boot'):
                continue
            for value in disk.get('licenses', []):
                if '1604' in value:
                    return '16.04'
                if '1404' in value:
                    return '14.04'
        return '14.04'
    return '14.04'


def get_volumes(instance):
    """Returns all the volumes of an instance."""
    if instance.cloud == 'aws':
        client = boto3.session.Session().client('ec2', instance.region)
        devices = client.describe_instance_attribute(
            InstanceId=instance.id, Attribute='blockDeviceMapping').get('BlockDeviceMappings', [])
        volumes = client.describe_volumes(VolumeIds=[device['Ebs']['VolumeId']
                                                     for device in devices if device.get('Ebs', {}).get('VolumeId')]).get('Volumes', [])
        return {volume['Attachments'][0]['Device']: {'size': volume['Size'], 'volume_type': volume['VolumeType']} for volume in volumes}
    if instance.cloud == 'gcp':
        credentials = GoogleCredentials.get_application_default()
        compute = discovery.build('compute', 'v1', credentials=credentials)
        volumes = {}
        for disk in compute.instances().get(instance=instance.id,
                                            zone=instance.zone,
                                            project=instance.project).execute()['disks']:
            index = disk['index']
            name = disk['deviceName'] if disk['deviceName'] not in [u'persistent-disk-0', 'boot'] else instance.id
            if 'local-ssd' in disk['deviceName']:
                size = 375.0
                disk_type = 'local-ssd'
            else:
                size = float(disk.get('diskSizeGb', 0.))
                disk_type = 'pd-ssd'
            volumes[index] = {'size': size,
                              'type': disk['type'],
                              'deviceName': disk['deviceName'],
                              'interface': disk['interface'],
                              'diskType': disk_type}
        return volumes
    raise ValueError('Unknown cloud %s' % instance.cloud)


def get_persistent_address(instance):
    """Returns the public ip address of an instance."""
    if instance.cloud == 'aws':
        client = boto3.client('ec2', instance.region)
        try:
            client.describe_addresses(PublicIps=[instance.ip_address])
            return instance.ip_address
        except botocore.client.ClientError as exc:
            if exc.response.get('Error', {}).get('Code') != 'InvalidAddress.NotFound':
                raise
            # Address is not public
            return None
    if instance.cloud == 'gcp':
        credentials = GoogleCredentials.get_application_default()
        compute = discovery.build('compute', 'v1', credentials=credentials)
        try:
            return compute.addresses().get(address=instance.name, project=instance.project, region=instance.region).execute()['address']
        except errors.HttpError as exc:
            if 'was not found' in str(exc):
                return None
            raise
    raise ValueError('Unknown cloud %s' % instance.cloud)


def get_image(instance):
    if instance.cloud == 'gcp':
        if instance.autoscaling_group is None:
            return None
        compute = discovery.build('compute', 'v1')
        instance_template = compute.instanceTemplates().get(project=instance.project,
                                                            instanceTemplate=instance.tags['instance-template']
                                                            .rsplit('/', 1)[1]).execute()

        return instance_template['properties']['disks'][0]['initializeParams']['sourceImage'].split('/')[-1]
    elif instance.cloud == 'aws':
        ec2 = boto3.client('ec2', instance.region)
        return ec2.describe_instances(InstanceIds=[instance.id]).get(u'Reservations', [])[0]['Instances'][0].get('ImageId', None)
    else:
        raise NotImplementedError('Unsupported cloud {}'.format(instance.cloud))
    return None


def get_cloud_from_zone(zone):
    for region in AWS_REGION_LIST:
        if region in zone:
            return 'aws'
    for region in GCP_REGION_LIST:
        if region in zone:
            return 'gcp'
    raise ValueError('Unknown zone %s' % zone)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('name', nargs='*', help='name to search')
    parser.add_argument('--by-id', action='store_true', default=False,
                        help='Deprecated- All searches are by id or name')
    parser.add_argument('-w', '--width', required=False, type=int, default=0, help='printed table width')
    parser.add_argument('-l', '--align', required=False, type=str, default='c', choices=ALIGN_OPTIONS,
                        help='align output {0}'.format(ALIGN_OPTIONS))
    parser.add_argument('-s', '--sort-by', nargs='*', required=False,
                        default=['cloud', 'name', 'region'], help='sort by (list of fields to sort by)')
    parser.add_argument('-f', '--fields', nargs='*', required=False, default=DEFAULT_SHOWN_FIELDS,
                        help='fields to output legal values are: {}'.format(INSTANCE_FILEDS))
    parser.add_argument('-P', '--projects', nargs='*', required=False, default=None,
                        help='gcp projects to search')
    parser.add_argument('--raw', action='store_true', default=False, help='raw ouput (no pretify)')
    parser.add_argument('-r', '--regions', default=None, nargs='*', help='Regions to search in')
    parser.add_argument('--strict', action='store_true', default=False,
                        help='search without automatic prefix and suffix *')
    parser.add_argument('--active', action='store_true', default=False, help='Only list active instances')
    parser.add_argument('--gevent', action='store_true', default=False, help='Use gevent')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Quiet (no headlines)')
    parser.add_argument('--clouds', nargs='*', default=SUPPORTED_CLOUDS,
                        help='clouds to search default: {}'.format(SUPPORTED_CLOUDS))

    args = parser.parse_args()
    if not args.name:
        name = sys.argv.pop(-1)
        args = parser.parse_args()
        args.name = [name]
    args.clouds = [cloud.lower() for cloud in args.clouds]

    return args


def main(args):
    if args.gevent:
        import gevent.monkey
        gevent.monkey.patch_all()
    if not args.strict:
        args.name = ['*' + name + '*' for name in args.name]
    table = Texttable(args.width)
    table.set_cols_dtype(['t'] * len(args.fields))
    table.set_cols_align([args.align] * len(args.fields))
    if not args.quiet:
        table.add_row([pretify_field(field) for field in args.fields])
    table.set_deco(0)
    instances = []
    for required_name in args.name:
        instances.extend(get_instances_by_id_or_name(required_name, projects=args.projects,
                                                     raw=args.raw, regions=args.regions, clouds=args.clouds))
    if args.active:
        instances = [instance for instance in instances if instance.state == 'running']
    if not instances:
        sys.exit(1)
    instances.sort(key=lambda instance: [getattr(instance, field) for field in args.sort_by])
    table.add_rows([[getattr(instance, field) for field in args.fields]
                    for instance in instances], header=False)
    print(table.draw())


if __name__ == '__main__':
    main(parse_args())
