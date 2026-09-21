# GitHub: cuenta, vuestro repositorio y cómo trabajar desde la DGX

Vuestro código vive en GitHub, no en la DGX. La DGX no tiene copias de seguridad, las
sesiones se acaban, y el repositorio público es además lo que vais a enseñar al terminar. Así
que lo primero es tener una cuenta, crear vuestro repositorio a partir de la plantilla, y
dejar la DGX configurada para hacer `push` sin dolor. Son quince minutos que os ahorran
muchos disgustos.

## 1. La cuenta

Si no tenéis cuenta, la creáis en [github.com/signup](https://github.com/signup). Un consejo:
usad un nombre de usuario que no os dé vergüenza poner en un currículum dentro de cinco años.
Podéis registraros con el correo personal y añadir después el de Comillas en *Settings →
Emails*; con el correo de la universidad verificado podéis pedir el
[GitHub Student Developer Pack](https://education.github.com/pack), que entre otras cosas da
GitHub Pro y Copilot gratis mientras seáis estudiantes.

Activad la verificación en dos pasos (*Settings → Password and authentication*). GitHub la
exige para casi todo y os la va a pedir tarde o temprano.

## 2. Vuestro repositorio a partir de la plantilla

Este repositorio está marcado como plantilla. No lo forkéis: cread el vuestro.

1. Entrad en [github.com/kendrickcetina/dgm-arca](https://github.com/kendrickcetina/dgm-arca).
2. Pulsad el botón verde **Use this template → Create a new repository**.
3. Elegid como propietario a uno de los miembros del equipo, poned un nombre (por ejemplo
   `arca-<vuestro-tema>`), una descripción de una línea, y dejadlo **público**. Es un
   proyecto de portfolio; que se vea.
4. Añadid al resto del equipo en *Settings → Collaborators* con permiso de escritura.

Con eso tenéis una copia limpia del enunciado, con vuestro nombre y sin historial ajeno. A
partir de aquí, el repositorio que importa es el vuestro.

## 3. Clonar en la DGX

Clonar un repositorio público no necesita ninguna credencial. Desde la terminal de
code-server, dentro de vuestra carpeta de trabajo:

```bash
cd ~/clusters/dgx
git clone https://github.com/<vuestro-usuario>/<vuestro-repo>.git
cd <vuestro-repo>
source smoke/dgx_env.sh
```

Hasta aquí podéis trabajar, entrenar y probar. Lo que no podéis todavía es hacer `push`.

## 4. Hacer `push` desde la DGX

Para escribir en GitHub hace falta autenticarse, y desde la DGX la forma más sencilla y
segura es un **token de acceso personal de grano fino**, limitado a vuestro repositorio y con
fecha de caducidad. La contraseña de GitHub no sirve para `git push` desde hace años.

### Crear el token (una vez, desde vuestro ordenador)

1. En GitHub: *Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token*.
2. Nombre: `dgx-arca`. Caducidad: el final del cuatrimestre. Si el repositorio es del equipo,
   en *Resource owner* elegid a quien lo tenga.
3. En *Repository access* marcad **Only select repositories** y elegid solo el vuestro.
4. En *Permissions → Repository permissions*, poned **Contents: Read and write**. Nada más.
5. Generad el token y copiadlo. Solo se muestra una vez.

Un token así, si se filtra, solo puede tocar ese repositorio y solo hasta que caduque. Por
eso lo preferimos a un token clásico con permiso `repo` sobre toda vuestra cuenta.

### Configurar git en la DGX (una vez)

`smoke/dgx_env.sh` hace que la configuración global de git viva en
`~/clusters/dgx/.gitconfig`, que sí se conserva entre sesiones. Con el script cargado:

```bash
git config --global user.name "Nombre Apellido"
git config --global user.email "vuestro-correo@ejemplo.com"
git config --global credential.helper "store --file $HOME/clusters/dgx/.git-credentials"
```

La primera vez que hagáis `push`, git os pedirá usuario y contraseña. Escribid vuestro
**usuario de GitHub** y, como contraseña, **el token**. Se guarda en
`~/clusters/dgx/.git-credentials` y no os lo vuelve a pedir.

```bash
git add -A
git commit -m "Primer commit desde la DGX"
git push
```

### Lo que no hay que hacer

- No pongáis el token en el código, en `.env`, en el README ni en `EXPERIMENTS.md`. Si un
  token acaba en un commit, GitHub lo revoca automáticamente y os avisa; revocadlo vosotros
  también y generad otro.
- No uséis vuestra contraseña de GitHub en la terminal.
- El fichero `.git-credentials` guarda el token en claro dentro de vuestra carpeta personal
  de la DGX. Es aceptable porque el token está limitado y caduca, pero cuando termine la
  práctica, borrad el fichero y revocad el token en GitHub.

### Alternativa: clave SSH

Si preferís SSH, generad una clave dentro de vuestra carpeta de trabajo y añadid la parte
pública en GitHub (*Settings → SSH and GPG keys*):

```bash
mkdir -p ~/clusters/dgx/.ssh && chmod 700 ~/clusters/dgx/.ssh
ssh-keygen -t ed25519 -f ~/clusters/dgx/.ssh/id_ed25519 -C "dgx-arca"
cat ~/clusters/dgx/.ssh/id_ed25519.pub
git config --global core.sshCommand "ssh -i $HOME/clusters/dgx/.ssh/id_ed25519 -o IdentitiesOnly=yes"
git remote set-url origin git@github.com:<vuestro-usuario>/<vuestro-repo>.git
```

Funciona igual de bien, pero depende de que el clúster permita conexiones de salida por el
puerto 22. Si `ssh -T git@github.com` se queda colgado, volved al token.

## 5. Hábitos que os van a salvar

- `git push` al terminar cada sesión de trabajo. Lo que no está en GitHub no existe.
- Los pesos, los índices y los datos grandes **no** van al repositorio: ya están en
  `.gitignore`. Los adaptadores LoRA pesan poco, subidlos a
  [Hugging Face Hub](https://huggingface.co/) con `huggingface-cli upload`, o descargadlos.
- Commits pequeños con mensajes que digan qué cambia. Dentro de dos meses os lo agradeceréis
  al escribir el informe, y a mí me sirve para ver cómo habéis trabajado.
- Cuando actualice el enunciado, os avisaré. Como vuestro repositorio viene de una plantilla y
  no es un fork, los cambios no llegan solos; os diré qué ficheros copiar.
