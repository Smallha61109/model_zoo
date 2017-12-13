wget --user=icalrobot --password='kvu4xu,4robot' ftp://ftp.ical.tw/tmp/data.tar.gz
wget --user=icalrobot --password='kvu4xu,4robot' ftp://ftp.ical.tw/tmp/rgbd_dataset_freiburg1_rpy.tgz
wget --user=icalrobot --password='kvu4xu,4robot' ftp://ftp.ical.tw/tmp/rgbd_dataset_freiburg2_rpy.tgz
wget --user=icalrobot --password='kvu4xu,4robot' ftp://ftp.ical.tw/tmp/rgbd_dataset_freiburg2_xyz.tgz
tar zxvf data.tar.gz
tar zxvf rgbd_dataset_freiburg2_xyz.tgz
tar zxvf rgbd_dataset_freiburg2_rpy.tgz
tar zxvf rgbd_dataset_freiburg1_rpy.tgz
mv data rgbd_dataset_freiburg1_xyz
rm data.tar.gz
rm rgbd_dataset_freiburg2_rpy.tgz
rm rgbd_dataset_freiburg2_xyz.tgz
rm rgbd_dataset_freiburg1_rpy.tgz
