import os, sys, math, random, argparse, subprocess, numpy as np
import bpy
import blender_util as bu

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str,   default='image.hdr', help='filename of rendered image')
parser.add_argument('--rngseed',  type=int,   default=-1,   help='random number generator seed')
parser.add_argument('--nobjects', type=int,   default=20,   help='number of objects in the scene')
parser.add_argument('--objtypes', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5])
parser.add_argument('--textypes', type=int, nargs='*', default=[0, 1, 2])
parser.add_argument('--imtexrange', type=int, default=1)
parser.add_argument('--prand', type=int, default=0)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--engine', type=str, default='eevee')

try:
    argv = sys.argv[ sys.argv.index('--')+1: ]
except:
    argv = ''
args = parser.parse_args(argv)

# generate a random xyz location
def rxyz( scale = 1 ):
    return( tuple( [ random.uniform( 0, scale ) for i in range(3) ] ) )

# generate a random greyscale rgb value
def rrgb( min=0.05, max=0.80, grey=True ):
    if grey:
        return 3*(random.uniform(min,max),) + (1.0,)
    else:
        return tuple( random.uniform(min,max) for i in range(3) ) + (1.0,)

# seed rng
if args.rngseed >= 0:
    random.seed(args.rngseed)
else:
    random.seed()

# delete cube
bpy.data.objects.remove( bpy.data.objects['Cube'] )

# set rendering parameters
render = bpy.context.scene.render
render.resolution_x = render.resolution_y = args.resolution
render.resolution_percentage = 100

# initialize intrinsic rendering manager
irm = bu.Intrinsic(hdr=True)
irm.reflectance = True
irm.normals = False
irm.depth = False
irm.objectid = False
irm.max_bounces = 0

# initialize camera manager
camera = bu.Camera( distance=10.0, fov=50.0, bg=None )
camera.azimuth = random.uniform( 0.0, 90.0 )
camera.elevation = random.uniform( 20.0, 60.0 )
camera.camera.location[1] = random.uniform( -0.5, 0.5 )
camera.camera.location[2] = random.uniform( -0.5, 0.5 )

# initialize lighting manager
light = bu.Light()
light.ambient = 1.0, 1.0
if args.prand:
    light.angle = random.uniform(0.0,5.0) if random.uniform(0.0,1.0)<0.75 else 0.0
else:
    light.angle = 2.0
light.point = 1.0, random.uniform(7.3/math.sqrt(10.0), 7.3*math.sqrt(10.0))
light.azimuth = random.uniform(10.0, 80.0)
light.elevation = random.uniform(50.0, 90.0)

def plane( name='Plane', rgb=(0.4,0.4,0.4,1.0) ):
    bpy.ops.mesh.primitive_plane_add( size=20, location=(0,0,0), scale=(1,1,1) )
    p = bpy.context.active_object
    p.name = name
    m = bpy.data.materials.new( name+'Material' )
    m.diffuse_color = bu.expand_rgb( rgb )
    m.specular_intensity = 0.0
    p.data.materials.append( m )
    return p

p = plane( 'Floor', rgb=rrgb() )
p = plane( 'LeftWall', rgb=rrgb() )
p.rotation_euler[0] = math.radians(-90.0)
p.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0 ) )
p = plane( 'RightWall', rgb=rrgb() )
p.rotation_euler[1] = math.radians(90.0)
p.rotation_euler[2] = math.radians( random.uniform(-20.0,20.0) )

# create objects
for i in range(args.nobjects):

    # create object
    location = rxyz(5)
    size = random.uniform(0.4,1.5)
    rotation = tuple( random.uniform(0,2*math.pi) for i in range(3) )
    shape = random.choice(args.objtypes)
    if shape==0:
        bpy.ops.surface.primitive_nurbs_surface_sphere_add( location=location, radius=size/2 )
    elif shape==1:
        bpy.ops.mesh.primitive_cube_add( location=location, size=size, rotation=rotation )
    elif shape==2:
        bpy.ops.mesh.primitive_ico_sphere_add( location=location, radius=size/2, rotation=rotation, subdivisions=1 )
    elif shape==3:
        bpy.ops.mesh.primitive_cylinder_add( location=location, radius=size/2, depth=size, rotation=rotation, vertices=128 )
    elif shape==4:
        bpy.ops.mesh.primitive_cone_add( location=location, radius1=size/2, radius2=0, depth=size, rotation=rotation, vertices=128 )
    elif shape==5:
        bpy.ops.mesh.primitive_torus_add( location=location, major_radius=0.35*size, minor_radius=0.15*size, rotation=rotation, major_segments=128, minor_segments=128 )
    obj = bpy.context.active_object
    obj.name = 'object_%06d' % ( i + 1 )

    # create material
    name= 'object_%06d_material' % ( i + 1 )
    texture = random.choice(args.textypes)
    if texture==0:
        bu.diffuse( obj=obj, rgb=rrgb(), name=name )
    elif texture==1:
        bu.voronoi( obj=obj, scale=random.uniform(1.0,10.0), name=name )
    elif texture==2:
        bu.noise( obj=obj, name=name )
    elif texture==3:
        if args.imtexrange==1:
            fname = os.path.join(os.environ['SLURM_TMPDIR'], f'imtex/rgb_{random.randint(1,1000):06d}.tif')
        elif args.imtexrange==2:
            fname = os.path.join(os.environ['SLURM_TMPDIR'], f'imtex/rgb_{random.randint(1,50000):06d}.tif')
        bu.image( obj=obj, fname=fname, name=name )

    obj.pass_index = i+1
    obj.pass_index += 100 * texture

# render
irm.engine = args.engine 
irm.filename = os.path.splitext(args.filename)[0]
irm.render()

