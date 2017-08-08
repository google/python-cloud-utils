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

"""list instances from AWS, and GCP."""
import sys

from argparse import ArgumentParser
import boto.ec2
import boto3
import botocore
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import parse as parse_date
from functools import partial
from pytz import UTC
import re
from texttable import Texttable

try:
    from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
    import google.auth.exceptions  # pylint: disable=g-import-not-at-top
    from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
    from googleapiclient import errors  # pylint: disable=g-import-not-at-top
    import googleapiclient.errors  # pylint: disable=g-import-not-at-top
except ImportError:
    print "Warning, google-api-python-client or google-auth not installed, can't connect to GCP instances."

try:
    from gevent.pool import Pool  # pylint: disable=g-import-not-at-top
    import gevent.monkey  # pylint: disable=g-import-not-at-top

    gevent.monkey.patch_all()
except ImportError:
    gevent = None

AUTOSCALING_GROUP_TAG = u'aws:autoscaling:groupName'
AWS_REGION_LIST = ['us-east-1', 'eu-west-1', 'us-west-2',
                   'sa-east-1', 'ap-southeast-1', 'eu-central-1']
GCP_REGION_LIST = ['us-east1', 'asia-east1', 'asia-northeast1', 'asia-southeast1', 'europe-west1', 'us-central1',
                   'us-west1']
INSTANCE_FILEDS = ['cloud', 'created', 'name', 'region', 'id', 'state', 'ip_address', 'type', 'autoscaling_group',
                   'iam_or_service_account', 'public_dns_name', 'private_ip_address', 'project', 'zone',
                   'security_groups',
                   'tags', 'vpc_id']
DEFAULT_SHOWN_FIELDS = ['project', 'name', 'zone', 'id', 'state', 'ip_address', 'type', 'created',
                        'autoscaling_group', 'iam_or_service_account', 'private_ip_address']
GCP_INSTANCE_TYPE_DICT = {'standard': 'std', 'highmem': 'mem', 'n1-': '', 'highcpu': 'cpu', 'custom': 'ctm'}
ALIGN_OPTIONS = ['l', 'c', 'r']
Instance = namedtuple('Instance', INSTANCE_FILEDS)


def pretify_field(field):
    return ' '.join([word[0].upper() + word[1:] for word in field.split('_')])


def aws_get_instances_by_name(region, name, raw=True):
    """Get all instances in a region matching a name (with wildcards)."""
    return _aws_get_instance_by_tag(region, name, 'tag:Name', raw)


def aws_get_instances_by_autoscaling_group_in_region(region, name, raw=True):
    return _aws_get_instance_by_tag(region, name, 'tag:aws:autoscaling:groupName', raw)


def datetime_to_str(date):
    return date.astimezone(UTC).isoformat().split('+')[0].split('.')[0].replace('T', ' ')


def _aws_get_instance_by_tag(region, name, tag, raw):
    """Get all instances matching a tag."""
    con = boto.ec2.connect_to_region(region)
    if not con:
        return []
    matching_reservations = con.get_all_instances(filters={tag: '{}'.format(name)})
    if not matching_reservations:
        return []
    instances = []
    [[instances.append(Instance(cloud='aws',  # pylint: disable=expression-not-assigned
                                zone=instance.placement,
                                region=region,
                                id=instance.id,
                                name=instance.tags.get('Name', 'Empty'),
                                state=instance.state,
                                type=instance.instance_type if raw else instance.instance_type.replace('xlarge', 'xl'),
                                autoscaling_group=instance.tags.get(AUTOSCALING_GROUP_TAG, None),
                                public_dns_name=instance.public_dns_name,
                                ip_address=instance.ip_address,
                                iam_or_service_account=instance.instance_profile['arn'].split('/')[
                                    1] if instance.instance_profile else None,
                                private_ip_address=instance.private_ip_address,
                                project='aws',
                                security_groups=[group.name for group in instance.groups],
                                tags=instance.tags,
                                vpc_id=instance.vpc_id,
                                created=datetime_to_str(parse_date(instance.launch_time))))
      for instance in reservation.instances] for reservation in matching_reservations if reservation]
    return instances


COMPUTE = None


def _get_compute_api(credentials):
    try:
        global COMPUTE
        if not COMPUTE:
            credentials = credentials or GoogleCredentials.get_application_default()
            COMPUTE = discovery.build('compute', 'v1', credentials=credentials)
    except google.auth.exceptions.DefaultCredentialsError:
        return None
    return COMPUTE


def gcp_get_instances_by_label(project, name, raw=True, credentials=None):
    compute = _get_compute_api(credentials)
    if not compute:
        return []
    region_to_instances = compute.instances().aggregatedList(project=project,
                                                             filter='labels.name eq {name}'.format(
                                                                 name=name.replace('*', '.*'))).execute().get('items',
                                                                                                              [])
    return get_instace_object_from_gcp_list(project, raw, region_to_instances)


def gcp_get_instances_by_name(project, name, raw=True, credentials=None):
    """Get instances in a GCP region matching name (with * wildcards)."""
    compute = _get_compute_api(credentials)
    if not compute:
        return []
    try:
        region_to_instances = COMPUTE.instances().aggregatedList(project=project,
                                                                 filter='name eq {name}'.format(
                                                                     name=name.replace('*', '.*'))).execute().get(
            'items', [])
    except googleapiclient.errors.HttpError as exc:
        if exc.resp['status'] in ['403', '404']:
            return []
        raise
    return get_instace_object_from_gcp_list(project, raw, region_to_instances)


def get_instace_object_from_gcp_list(project, raw, region_to_instances):
    matching_intsances = []
    for region, region_instances in region_to_instances.iteritems():
        for matching_instance in region_instances.get('instances', []):
            created_by = [item['value'] for item in matching_instance.get('metadata', {}).get('items', []) if
                          item['key'] == 'created-by']
            if created_by and 'instanceGroupManagers' in created_by[0]:
                if raw:
                    autoscaling_group = created_by[0]
                else:
                    autoscaling_group = created_by[0].split('/')[-1]
            else:
                autoscaling_group = None
            public_ip = None
            if raw:
                iam_or_service_account = ','.join(
                    [account['email'] for account in matching_instance['serviceAccounts']])
                instance_type = matching_instance['machineType'].split('/')[-1]
            else:
                iam_or_service_account = ','.join(
                    [account['email'].split('@')[0] for account in matching_instance['serviceAccounts']])
                instance_type = matching_instance['machineType'].split('/')[-1]
                for raw_type, pretty_type in GCP_INSTANCE_TYPE_DICT.iteritems():
                    instance_type = instance_type.replace(raw_type, pretty_type)
            for interface in matching_instance['networkInterfaces']:
                if 'accessConfigs' in interface and interface['accessConfigs'] and interface['accessConfigs'][0].get(
                        'natIP'):
                    public_ip = interface['accessConfigs'][0]['natIP']
            tags = {metadata['key']: metadata['value'] for metadata in matching_instance['metadata'].get('items', [])}
            tags.update(matching_instance.get('labels', {}))
            tags[None] = matching_instance['tags'].get('items', [])

            creation_datetime = datetime_to_str(parse_date(matching_instance['creationTimestamp']))
            matching_intsances.append(Instance(cloud='gcp',
                                               zone=region.replace('zones/', ''),
                                               region=region.replace('zones/', '')[:-2],
                                               id=str(matching_instance['id']),
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
                                               vpc_id=None))
    return matching_intsances


def aws_get_instances_by_id(region, instance_id, raw=True):
    """Returns instances mathing an id."""
    con = boto.ec2.connect_to_region(region)
    if not con:
        return []
    matching_reservations = con.get_all_instances(filters={'instance-id': instance_id})
    matching_instances = []
    [[matching_instances.append(Instance(cloud='aws',  # pylint: disable=expression-not-assigned
                                         zone=instance.placement,
                                         region=region,
                                         id=instance.id,
                                         name=instance.tags.get('Name', 'Empty'),
                                         state=instance.state,
                                         type=instance.instance_type,
                                         autoscaling_group=instance.tags.get(AUTOSCALING_GROUP_TAG, None),
                                         public_dns_name=instance.public_dns_name,
                                         ip_address=instance.ip_address,
                                         iam_or_service_account=instance.instance_profile[
                                             'arn'].split('/')[1] if instance.instance_profile else None,
                                         private_ip_address=instance.private_ip_address,
                                         project='aws',
                                         security_groups=[group.name for group in instance.groups],
                                         tags=instance.tags,
                                         created=datetime_to_str(parse_date(instance.launch_time)),
                                         vpc_id=instance.vpc_id))
      for instance in reservation.instances] for reservation in matching_reservations if reservation]
    return matching_instances


def get_instances_by_id(instance_id, regions=None, projects=None, raw=True,
                        sort_by_order=('cloud', 'name')):  # pylint: disable=unused-argument
    with ThreadPoolExecutor(max_workers=len(AWS_REGION_LIST)) as executor:
        results = executor.map(partial(aws_get_instances_by_id, instance_id=instance_id),
                               AWS_REGION_LIST)
    matching_instances = []
    for result in results:
        matching_instances.extend(result)
    if regions:
        matching_instances = [instance for instance in matching_instances if instance.region in regions]
    return matching_instances


def _get_instances_using_threads(name, projects, raw, credentials):
    with ThreadPoolExecutor(max_workers=2 * len(AWS_REGION_LIST) + 2 * len(projects)) as executor:
        aws_name_results = executor.map(partial(aws_get_instances_by_name, name=name, raw=raw),
                                        AWS_REGION_LIST)
        aws_autoscaling_results = executor.map(
            partial(aws_get_instances_by_autoscaling_group_in_region, name=name, raw=raw),
            AWS_REGION_LIST)
        if credentials:
            gcp_name_results = executor.map(
                partial(gcp_get_instances_by_name, name=name, raw=raw, credentials=credentials),
                projects)
            gcp_label_results = executor.map(
                partial(gcp_get_instances_by_label, name=name, raw=raw, credentials=credentials),
                projects)
        else:
            gcp_name_results = gcp_label_results = []
    matching_instances = []
    for instances in [result for result in aws_name_results] + [result for result in gcp_name_results] + \
            [result for result in gcp_label_results] + [result for result in aws_autoscaling_results]:
        # for instances in results:
        matching_instances.extend(instances)
    return matching_instances


def _get_instances_using_gevent(name, projects, raw, credentials):
    pool = Pool(size=None)
    matching_instances = []
    aws_name_results = pool.map_async(partial(aws_get_instances_by_name, name=name, raw=raw), AWS_REGION_LIST,
                                      callback=lambda results: [matching_instances.extend(instances) for instances in
                                                                results])
    aws_autoscaling_results = pool.map_async(
        partial(aws_get_instances_by_autoscaling_group_in_region, name=name, raw=raw), AWS_REGION_LIST,
        callback=lambda results: [matching_instances.extend(instances) for instances in results])
    gcp_name_results = pool.map_async(partial(gcp_get_instances_by_name, name=name, raw=raw, credentials=credentials),
                                      projects,
                                      callback=lambda results: [matching_instances.extend(instances) for instances in
                                                                results])
    gcp_label_results = pool.map_async(partial(gcp_get_instances_by_label, name=name, raw=raw, credentials=credentials),
                                       projects,
                                       callback=lambda results: [matching_instances.extend(instances) for instances in
                                                                 results])
    aws_name_results.join()
    aws_autoscaling_results.join()
    gcp_name_results.join()
    gcp_label_results.join()
    return matching_instances


def get_instances_by_name(name, sort_by_order=('cloud', 'name'), projects=None, raw=True, regions=None,
                          gcp_credentials=None):
    """Get intsances from GCP and AWS by name."""
    try:
        credentials = gcp_credentials or GoogleCredentials.get_application_default()
        resourcemanager = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
        projects = projects if projects is not None else [project['projectId']
                                                          for project in
                                                          resourcemanager.projects().list().execute().get('projects',
                                                                                                          [])]
    except google.auth.exceptions.DefaultCredentialsError:
        print "Error getting google application credentials. " \
              "Can't connect to google instances. Run 'gcloud auth application-default login'"
        projects = []
        credentials = None
    if gevent:
        matching_instances = _get_instances_using_gevent(name, projects, raw, credentials)
    else:
        matching_instances = _get_instances_using_threads(name, projects, raw, credentials)
    if regions:
        matching_instances = [instance for instance in matching_instances if instance.region in regions]
    matching_instances.sort(key=lambda instance: [getattr(instance, field) for field in sort_by_order])
    return matching_instances


def get_instances_by_id_or_name(identifier):
    id_regex = re.compile('^i?-?(?P<id>[0-9a-f]{17}|[0-9a-f]{8})$')
    if id_regex.match(identifier):
        matching_instances = get_instances_by_id(identifier)
        if matching_instances:
            return matching_instances[0]
    return get_instances_by_name(identifier)


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
        client = boto3.client('ec2', instance.region)
        devices = client.describe_instance_attribute(InstanceId=instance.id, Attribute='blockDeviceMapping').get(
            'BlockDeviceMappings', [])
        volumes = client.describe_volumes(VolumeIds=[device['Ebs']['VolumeId']
                                                     for device in devices if
                                                     device.get('Ebs', {}).get('VolumeId')]).get('Volumes', [])
        return {volume['Attachments'][0]['Device']: {'size': volume['Size'], 'volume_type': volume['VolumeType']} for
                volume in volumes}
    if instance.cloud == 'gcp':
        credentials = GoogleCredentials.get_application_default()
        compute = discovery.build('compute', 'v1', credentials=credentials)
        volumes = {}
        for disk in compute.instances().get(instance=instance.name,
                                            zone=instance.zone,
                                            project=instance.project).execute()['disks']:
            index = disk['index']
            name = disk['deviceName'] if disk['deviceName'] != u'persistent-disk-0' else instance.name
            if 'local-ssd' in disk['deviceName']:
                size = 375.0
            if 'local-ssd' in disk['deviceName']:
                size = 375.0
                disk_type = 'local-ssd'
            else:
                disk_data = compute.disks().get(disk=name,
                                                zone=instance.zone,
                                                project=instance.project).execute()
                size = float(disk_data['sizeGb'])
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
            return \
            compute.addresses().get(address=instance.name, project=instance.project, region=instance.region).execute()[
                'address']
        except errors.HttpError as exc:
            if 'was not found' in str(exc):
                return None
            raise
    raise ValueError('Unknown cloud %s' % instance.cloud)


def get_cloud_from_zone(zone):
    for region in AWS_REGION_LIST:
        if region in zone:
            return 'aws'
    for region in GCP_REGION_LIST:
        if region in zone:
            return 'gcp'
    raise ValueError('Unknown zone %s' % zone)


def main():
    parser = ArgumentParser()
    parser.add_argument('name', nargs='+', help='name to search')
    parser.add_argument('--by-id', action='store_true', default=False, help='Search instance by id')
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
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Quiet (no headlines)')
    args = parser.parse_args()
    if not args.strict:
        args.name = ['*' + name + '*' for name in args.name]
    table = Texttable(args.width)
    table.set_cols_dtype(['t'] * len(args.fields))
    table.set_cols_align([args.align] * len(args.fields))
    if not args.quiet:
        table.add_row([pretify_field(field) for field in args.fields])
    table.set_deco(0)
    instances = []
    search_method = get_instances_by_id if args.by_id else get_instances_by_name
    for required_name in args.name:
        instances.extend(search_method(required_name, projects=args.projects, raw=args.raw, regions=args.regions))
    if args.active:
        instances = [instance for instance in instances if instance.state == 'running']
    if not instances:
        sys.exit(1)
    instances.sort(key=lambda instance: [getattr(instance, field) for field in args.sort_by])
    table.add_rows([[getattr(instance, field) for field in args.fields]
                    for instance in instances], header=False)
    print table.draw()


if __name__ == '__main__':
    main()
