# :)

IGNORED_EXTENSIONS = [
    # images
    'mng', 'pct', 'bmp', 'gif', 'jpg', 'jpeg', 'png', 'pst', 'psp', 'tif',
    'tiff', 'ai', 'drw', 'dxf', 'eps', 'ps', 'svg',

    # audio
    'mp3', 'wma', 'ogg', 'wav', 'ra', 'aac', 'mid', 'au', 'aiff',

    # video
    '3gp', 'asf', 'asx', 'avi', 'mov', 'mp4', 'mpg', 'qt', 'rm', 'swf', 'wmv',
    'm4a',

    # office suites
    'xls', 'xlsx', 'ppt', 'pptx', 'doc', 'docx', 'odt', 'ods', 'odg', 'odp', 'key', 'bib',

    # other
    'css', 'pdf', 'exe', 'bin', 'rss', 'zip', 'rar', 'gz', 'txt',

    # code
    'c', 'h', 'java','dll', 'tcl'
]

def is_valid(url):
    split = url.split(".")
    extension = split[-1]
    return not extension in IGNORED_EXTENSIONS
